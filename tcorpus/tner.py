from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd


def linearize_labels(label):
    """
    Convert label information from flair object to pd Series
    
    :param label: instance of flair label
    :return: pd series of label text, start pos, end pos, label, confidence
    """
    if pd.isna(label):
        return pd.Series([])
    else:
        return pd.Series([label['text'], label['start_pos'], label['end_pos'], label['labels'][0]['value'], label['labels'][0]['confidence']])


def ner(sentences, flair_model, text_col='text', keep_cols=None):
    """
    Perform named entity recognition.
    
    :param sentences: data frame with sentences
    :param flair_model: name of flair model
    :param text_col: column in sentences containing sentence text
    :param keep_cols: columns in texts to keep in return data frame
    :return: data frame of named entities in sentences
    """

    if keep_cols is None:
        keep_cols = []
    
    if not set(keep_cols).issubset(list(sentences.columns)):
        raise KeyError('at least one column in keep_cols is missing in sentences data frame')

    tagger = SequenceTagger.load(flair_model)
    flair_sentences = sentences[text_col].astype(str).apply(Sentence)
    tagger.predict(flair_sentences.to_list())
    entities = flair_sentences.apply(lambda sentence: [ span.to_dict() for span in sentence.get_spans('ner') ]).explode()
    entities = entities[entities.notna()].apply(linearize_labels)
    entities.columns = ['entity_text', 'start_pos', 'end_pos', 'entity_type', 'confidence']
    entities = entities.join(sentences[keep_cols])
    entities.reset_index(names='sentence_id', inplace=True)
    return entities