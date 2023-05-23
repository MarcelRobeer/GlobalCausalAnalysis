import requests
import json
from datasets import ClassLabel, Dataset, Sequence, load_dataset

from gca.config import DATASET_NAME, FORCE_INFER, HUGGINGFACE_ACCOUNT
from gca.data.inferred_attributes import inferred_attributes
from gca.data.tasks import FAIRNESS_FEATURES, ROBUSTNESS_FEATURES, TASK_FEATURES
from gca.utils import huggingface_login


DEFAULT_DATA_NAME: str = f'{HUGGINGFACE_ACCOUNT}/{DATASET_NAME}'


def prepare_data():
    _dataset_raw = load_dataset('go_emotions', 'raw')
    _dataset = load_dataset('go_emotions', 'simplified')

    # All values that are unique by ID
    _unique_vals = _dataset_raw['train'].with_format('pandas')[:][['id', 'author', 'subreddit', 'created_utc']] \
                                        .groupby('id') \
                                        .first()

    # All values with multiple values (aggregated as list) by ID
    _list_vals = _dataset_raw['train'].with_format('pandas')[:][['id', 'rater_id']] \
                                      .groupby('id') \
                                      .agg(list)

    # Merge them into one overview
    _vals = _unique_vals.merge(_list_vals, left_index=True, right_index=True)

    # Get names of all features
    features = _dataset['train'].features
    for k, v in Dataset.from_pandas(_vals.reset_index(drop=True)).features.items():
        features[k] = v


    def add_columns(data: Dataset) -> Dataset:
        return Dataset.from_pandas(data.with_format('pandas')[:]
                                       .merge(_vals, left_on='id', right_on='id')
                                       .reset_index(drop=True),
                                   features=features)

    # Group fine-grained emotions (27 + 1) into high-level sentiments (4)
    sentiment_mapping = json.loads(requests.get('https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/sentiment_mapping.json').text)
    sentiment_mapping['neutral'] = ['neutral']
    sentiment_mapping = {_dataset['train'].features['labels'].feature.str2int(v): k
                         for k, l in sentiment_mapping.items()
                         for v in l}


    def get_highlevel_sentiment(dataset: Dataset) -> Dataset:
        def get_sentiment(batch):
            def unique_sentiment(example):
                return list(set(sentiment_mapping[label] for label in example))

            return {'sentiment_list': [unique_sentiment(example) for example in batch['labels']]}

        return dataset.with_format(None) \
            .map(get_sentiment, batched=True) \
            .cast_column('sentiment_list', Sequence(ClassLabel(names=['positive', 'neutral', 'negative', 'ambiguous']))) \
            .rename_column('labels', 'detailed_labels')


    # Remove any instances with multiple high-level sentiment values
    def unique_sentiment(dataset: Dataset) -> Dataset:
        feature = dataset.features['sentiment_list'].feature
        return dataset.filter(lambda x: len(x) == 1, input_columns='sentiment_list') \
            .map(lambda x: {'labels': x[0]}, input_columns='sentiment_list', remove_columns='sentiment_list') \
            .cast_column('labels', feature)

    for split in _dataset.keys():
        print(f'>>> Processing split "{split}"')
        _dataset[split] = add_columns(_dataset[split])
        _dataset[split] = inferred_attributes(_dataset[split])
        _dataset[split] = unique_sentiment(get_highlevel_sentiment(_dataset[split]))

    return _dataset


def print_descriptives(dataset: Dataset) -> None:
    import pandas as pd

    total_instances = len(dataset['train']) + len(dataset['test']) + len(dataset['validation'])
    print(f'Total instances: {total_instances}')
    for split in ['train', 'test', 'validation']:
        print(f'  > {split:<10} | {len(dataset[split]):<5} instances | percentage {len(dataset[split]) / total_instances:.0%}')

    label_distributions = pd.concat([dataset[split].with_format('pandas')['labels'].replace(dict(zip(range(4), dataset[split].features['labels']._int2str))).value_counts().rename(split, inplace=True).to_frame()
                                    for split in ['train', 'test', 'validation']], axis=1).loc[['positive', 'negative', 'ambiguous', 'neutral']]
    print(label_distributions)


def get_data(name: str = DEFAULT_DATA_NAME,
             force: bool = FORCE_INFER):
    import warnings

    if not force:
        try:
            if name.endswith('.csv'):
                # Local file
                from datasets.utils.file_utils import relative_to_absolute_path
                dataset = load_dataset('csv', data_files=relative_to_absolute_path(name))['train']
            else:
                # Online file
                huggingface_login()
                dataset = load_dataset(name)
        except Exception as e:
            warnings.warn(e)
            force = True

    if force:
        dataset = prepare_data()
        dataset.push_to_hub(name, private=True)
    return dataset


if __name__ == '__main__':
    huggingface_login()
    dataset = get_data(DEFAULT_DATA_NAME)
    print_descriptives(dataset)
