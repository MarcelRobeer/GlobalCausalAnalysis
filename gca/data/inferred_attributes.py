import emoji
import spacy
import textstat

from PassivePySrc.rules_for_all_passives import create_matcher


nlp = spacy.load('en_core_web_sm')
ruler = nlp.add_pipe('entity_ruler')
ruler.add_patterns([{'label': 'PERSON', 'pattern': '[NAME]'},
                    {'label': 'RELIGION', 'pattern': '[RELIGION]'}])
_, matcher = create_matcher(nlp=nlp)


def dict_apply(texts, fn):
    dicts = [fn(text) for text in texts]
    return {k: [dic[k] for dic in dicts] for k in dicts[0]}


def batchwise_apply(texts, fn):
    return [fn(text) for text in texts]


def inferred_attributes(dataset):
    import pandas as pd
    from gca.data.lexicon import get_scores_from_lookuptable, nrc_lexicon, genbit_lexicon, pos_neg_lexicon

    lookuptable = pd.concat([nrc_lexicon(), genbit_lexicon(), pos_neg_lexicon()])

    def attributes(batch):
        texts = [i for i in batch]
        docs = [doc for doc in nlp.pipe(texts)]
        return dict_apply(batch, lambda x: get_scores_from_lookuptable(x, lookuptable=lookuptable)) | \
               {'len_chr': batchwise_apply(texts, len),
                'len_tok': batchwise_apply(docs, lambda doc: sum(len(sent) for sent in doc.sents)),
                'len_snt': batchwise_apply(docs, lambda doc: len(list(doc.sents))),
                'entities': batchwise_apply(docs, lambda doc: [(ent.text, ent.label_) for ent in doc.ents]),
                'is_active': batchwise_apply(docs, lambda doc: len(matcher(doc)) == 0),
                'has_name': batchwise_apply(texts, lambda x: '[NAME]' in x),
                'has_religion': batchwise_apply(texts, lambda x: '[RELIGION]' in x),
                'has_emoji': batchwise_apply(texts, lambda x: any(emoji.is_emoji(c) for c in x)),
                'all_lower': batchwise_apply(texts, lambda x: str(x).islower()),
                'flesch_grade': batchwise_apply(texts, textstat.flesch_kincaid_grade)}

    return dataset.map(attributes, batched=True, input_columns='text')
