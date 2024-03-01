import syntok.segmenter as segmenter


def split_string(string, level):
    """
    split string into paragraphs, sentences, and tokens (token spacing, token value)
    
    :param split_string: nested list of paragraphs, sentences, and tokens (return value of split)
    :param level: split level ("paragraph", "sentence" or "token")
    :return: list of paragraphs (nested if level is sentence or token)
    """

    paragraphs = []
    for paragraph in segmenter.analyze(string):
        sentences = []
        for sentence in paragraph:
            if level == 'token':
                # only use token value (drop token spacings)
                tokens = [token.value for token in sentence]
                sentences.append(tokens)
            else:
                # rebuild sentence from token spacing and token values
                tokens = [char for token in sentence for char in (token.spacing, token.value)]
                sentences.append(''.join(tokens).strip())
        if level == 'paragraph':
            paragraphs.append(' '.join(sentences))
        else:
            paragraphs.append(sentences)
    return paragraphs    

def segment(texts, level, text_col='text', keep_cols=None):
    """
    create data frame of segments (paragraphs, sentences or words) from data frame of texts
    
    :param texts: data frame with texts
    :level: split level ("paragraph", "sentence" or "token")
    :param text_col: column containing text
    :param keep_cols: columns in texts to keep in return data frame
    :return: data frame with one per line per split level (paragraph, sentence or token)
    """
    
    if level not in ['paragraph', 'sentence', 'token']:
        raise ValueError('level must be one of "paragraph", "sentence" or "token"')
    
    if keep_cols is None:
        keep_cols = []
    
    if not set(keep_cols).issubset(list(texts.columns)):
        raise KeyError('at least one column in keep_cols is missing in texts data frame')
    
    df = texts[text_col].apply(split_string, level=level).explode().to_frame('paragraph')
    df = df.join(texts[keep_cols])
    df.reset_index(names='_'.join(text_col, 'id'), inplace=True)
    if level == 'paragraph':
        return df
    
    df = df.explode('paragraph')
    df.rename(columns={'paragraph':'sentence'}, inplace=True)
    df.reset_index(names='paragraph_id', inplace=True)
    if level == 'sentence':
        return df
    
    df = df.explode('sentence')
    df.rename(columns={'sentence':'token'}, inplace=True)
    df.reset_index(names='sentence_id', inplace=True)
    return df
