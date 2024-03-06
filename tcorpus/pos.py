# %%
from flair.nn import Classifier
from flair.data import Sentence
import pandas as pd


def init_tagger(flair_model_name):
    """
    Load tagger model into memory.

    :param flair_model_name: name of the flair model to use.
    :return: instance of flair Classifier
    """
    return Classifier.load(flair_model_name)


def pos(sentences, tagger, text_col="text", keep_cols=None):
    """
    Perform pos tagging.

    :param sentences: data frame with sentences
    :param tagger: pos tagger instance created by init_tagger(flair_model_name)
    :param text_col: column in sentences containing sentence text
    :param keep_cols: columns in texts to keep in return data frame
    :return: data frame of tokens with part-of-speech tags
    """

    sentences_copy = sentences.copy()
    sentences_copy = sentences_copy[sentences_copy[text_col].notna()]
    sentences_list = sentences_copy[text_col].to_list()

    flair_sentences = [Sentence(s) for s in sentences_list]
    for s in flair_sentences:
        tagger.predict(s)

    labels = [[lbl for lbl in s.get_labels("pos")] for s in flair_sentences]

    labels = pd.Series(labels, index=sentences_copy.index)
    labels = labels.explode()
    labels = labels[labels.notna()]

    pos = pd.DataFrame()
    pos["token"] = labels.apply(lambda x: x.data_point.text)
    pos["pos"] = labels.apply(lambda x: x.value)

    pos = pos.join(sentences_copy[keep_cols])
    pos.reset_index(names="sentence_id", inplace=True)
    return pos
