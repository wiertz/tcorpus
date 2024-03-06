from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd
from torch.cuda import empty_cache


def init_tagger(flair_model_name):
    """
    Load tagger model into memory.

    :param flair_model_name: name of the flair model to use.
    :return: instance of flair SequenceTagger
    """
    return SequenceTagger.load(flair_model_name)


def ner(sentences, tagger, text_col="text", keep_cols=None):
    """
    Perform named entity recognition.

    :param sentences: data frame with sentences
    :param tagger: tagger instance created by init_tagger(flair_model_name)
    :param text_col: column in sentences containing sentence text
    :param keep_cols: columns in texts to keep in return data frame
    :param max_sentence_len: maximum number of characters per sentence (avoid memory issues with flair)
    :return: data frame of named entities in sentences
    """

    if keep_cols is None:
        keep_cols = []

    if not set(keep_cols).issubset(list(sentences.columns)):
        raise KeyError(
            "at least one column in keep_cols is missing in sentences data frame"
        )

    # create copy to avoid manipulation of original df
    sentences_copy = sentences.copy()

    sentences_list = sentences_copy[text_col].astype(str).to_list()
    flair_sentences = [Sentence(s) for s in sentences_list]
    for s in flair_sentences:
        tagger.predict(s)

    labels = [[span.to_dict() for span in s.get_spans()] for s in flair_sentences]
    labels = pd.Series(labels, index=sentences_copy.index)
    labels = labels.explode()
    labels = labels[labels.notna()]

    entities = pd.DataFrame()

    # add columns with label information to entnties
    entities["text"] = labels.apply(lambda x: x["text"])
    entities["start_pos"] = labels.apply(lambda x: x["start_pos"])
    entities["end_pos"] = labels.apply(lambda x: x["end_pos"])
    entities["value"] = labels.apply(lambda x: x["labels"][0]["value"])
    entities["confidence"] = labels.apply(lambda x: x["labels"][0]["confidence"])
    entities = entities.join(sentences_copy[keep_cols])
    entities.reset_index(names="sentence_id", inplace=True)

    # empty cuda cache
    empty_cache()

    return entities
