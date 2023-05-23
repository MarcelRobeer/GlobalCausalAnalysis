from typing import Dict

import io
import numpy as np
import os
import pandas as pd
import zipfile

from lazy_load import lazy_func

from gca.utils import word_tokenizer


@lazy_func
def nrc_lexicon():
    def open_zip(path: str, innerpath: str) -> io.BytesIO:
        return zipfile.ZipFile(path).open(innerpath)


    # NRC Word-Emotion Association Lexicon (aka EmoLex), word emotions and valence/arousal/dominance scores
    if not os.path.exists('NRC-Suite-of-Sentiment-Emotion-Lexicons.zip'):
        raise FileNotFoundError('Download http://saifmohammad.com/WebDocs/Lexicons/NRC-Suite-of-Sentiment-Emotion-Lexicons.zip ',
                                'and place it in the same folder as your .ipynb file!')
    emotion_lexicon_file = open_zip('NRC-Suite-of-Sentiment-Emotion-Lexicons.zip',
                                    'NRC-Suite-of-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    nrc_emotion_lexicon = pd.read_csv(emotion_lexicon_file, sep='\t', header=None) \
                            .pivot(index=0, columns=1) \
                            .rename_axis(None, axis=0) \
                            .droplevel(0, axis=1) \
                            .rename_axis(None, axis=1) \
                            .drop(columns=['positive', 'negative'])

    vad_lexicon_file = open_zip('NRC-Suite-of-Sentiment-Emotion-Lexicons.zip',
                                'NRC-Suite-of-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt')
    nrc_vad_lexicon = pd.read_csv(vad_lexicon_file, sep='\t', header=None, names=['token', 'valence', 'arousal', 'dominance']) \
                        .set_index('token')

    NRC_LEXICON = pd.concat([nrc_emotion_lexicon, nrc_vad_lexicon], axis=1)
    NRC_LEXICON.columns = [f'NRC_{col}' for col in NRC_LEXICON.columns]
    return NRC_LEXICON

@lazy_func
def genbit_lexicon():
    # GenBit indicators of male/female/non-binary
    GENBIT_LEXICON = pd.concat([pd.read_csv(f'https://raw.githubusercontent.com/microsoft/responsible-ai-toolbox-genbit/main/genbit/gendered-word-lists/en/{gender}.txt', header=None).assign(**{f'{gender}_words': 1}).set_index(0).rename_axis(None)
                                for gender in ['male', 'female', 'non-binary']])
    return GENBIT_LEXICON


@lazy_func
def pos_neg_lexicon():
    positive_words = ['good', 'awesome', 'great', 'fantastic', 'wonderful', 'best', 'love', 'excellent']
    negative_words = ['bad', 'terrible', 'worst', 'sucks', 'awful', 'waste', 'boring', 'worse']

    # Positive and negative words
    POS_NEG_LEXICON = pd.DataFrame({'word': positive_words + negative_words,
                                    'positive_words': [1] * len(positive_words) + [0] * len(negative_words),
                                    'negative_words': [0] * len(positive_words) + [1] * len(negative_words)}) \
                        .set_index('word')
    return POS_NEG_LEXICON


AGG_MAP = {'NRC_anger': 'sum',
           'NRC_anticipation':'sum',
           'NRC_disgust': 'sum',
           'NRC_fear': 'sum',
           'NRC_joy': 'sum',
           'NRC_sadness': 'sum',
           'NRC_surprise': 'sum',
           'NRC_trust': 'sum',
           'NRC_valence': 'mean',
           'NRC_arousal': 'mean',
           'NRC_dominance': 'mean'}
AGG_MAP |= {col: 'sum' for col in genbit_lexicon().columns}
AGG_MAP |= {col: 'sum' for col in pos_neg_lexicon().columns}


def get_scores_from_lookuptable(input: str,
                                lookuptable: pd.DataFrame,
                                agg: Dict[str, str] = AGG_MAP,
                                unique: bool = True):
    tokens = (token if token in lookuptable.index else np.nan
              for token in word_tokenizer(input))
    try:
        all_scores = lookuptable.loc[list(tokens) if unique else set(tokens)]
        return {k: 0.0 if np.isnan(v) else v for k, v in all_scores.agg(agg).to_dict().items()}
    except KeyError:  # all np.nan
        return {k: 0 if isinstance(v, int) else 0.0 for k, v in lookuptable.dtypes.items()}

__all__ = ['nrc_lexicon', 'genbit_lexicon', 'pos_neg_lexicon', 'get_scores_from_lookuptable']
