from collections import defaultdict

DESCRIPTORS = defaultdict(lambda: '?',
                          {'len_chr': 'Document length in number of characters',
                           'len_tok': 'Document length in number of tokens',
                           'len_snt': 'Document length in number of sentences',
                           'has_name': 'If document contains the [NAME] token',
                           'has_emoji': 'If document contains a Unicode emoji',
                           'flesch_grade': 'Readability: Flesch-Kincaid reading grade of document FKGL = 0.39 (words/sentences) + 11.8 (syllables/words) - 15.59',
                           'is_active': 'If all sentences in document are in active voice',
                           'all_lower': 'If document is all lower characters',
                           'subreddit': 'HUMAN-ANNOTATED | Subreddit the document is sampled from',
                           'PRED_label': 'Predicted class label',
                          })
for emotion in ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']:
    DESCRIPTORS[f'NRC_{emotion}'] = f'Total tokens labelled with {emotion} according to NRC EmoLex'
for vad in ['valence', 'arousal', 'dominance']:
    DESCRIPTORS[f'NRC_{vad}'] = f'Mean {vad} score according to NRC EmoLex'
for pn in ['positive', 'negative']:
    DESCRIPTORS[f'{pn}_words'] = f'Number of {pn} words according to a short wordlist'
for gender in ['male', 'female', 'non-binary']:
    DESCRIPTORS[f'{gender}_words'] = f'Number of indicators for the {gender} gender according to the English GenBit wordlist'
for label in ['positive', 'negative', 'neutral', 'ambiguous']:
    DESCRIPTORS[f'PRED_{label}'] = f'Predicted class label for label {label} (one-vs-rest)'
