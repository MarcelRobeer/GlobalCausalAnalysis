from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TextClassificationPipeline, Trainer,
                          TrainingArguments)

from gca.config import DATASET_NAME, FORCE_TRAIN, GPU_BATCH_SIZE, HUGGINGFACE_ACCOUNT, MODEL, SEED


def format_name(model_name: str = MODEL) -> str:
    return f'{HUGGINGFACE_ACCOUNT}/{model_name.split("/")[-1]}_GoEmotions'


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro'
    )
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(dataset: Dataset, model_name: str = MODEL, seed: int = SEED):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_features(batch):
        inputs = tokenizer(batch['text'],
                           padding='max_length',
                           max_length=512,
                           truncation=True)
        inputs['labels'] = batch['labels']
        return inputs

    ds_training = dataset.with_transform(extract_features)

    classes = dataset['train'].features['labels']._str2int
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(classes),
        id2label={v: k for k, v in classes.items()},
        label2id=classes,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir='./test_trainer',
        push_to_hub=True,
        hub_private_repo=True,
        hub_model_id=format_name(model_name),
        remove_unused_columns=False,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        per_device_train_batch_size=GPU_BATCH_SIZE,
        per_device_eval_batch_size=GPU_BATCH_SIZE * 4,
        group_by_length=True,
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=AutoTokenizer.from_pretrained(model),
        train_dataset=ds_training['train'].shuffle(seed=seed),
        eval_dataset=ds_training['validation'],
        compute_metrics=compute_metrics,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics('train', train_results.metrics)
    trainer.save_metrics('train', train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(ds_training['test'])
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)
    return trainer, metrics


def apply(
        dataset: Dataset,
        pipeline: TextClassificationPipeline,
        dataset_name: str,
        model_name: str,
        ) -> Dataset:
    def apply_pipeline(batch):
        res = pipeline(batch)
        ret = {k: [dic[k] for dic in res] for k in res[0]}
        ret['PRED_label'] = ret.pop('label')
        ret['PRED_score'] = ret.pop('score')
        return ret

    ds_causal = dataset['test'].map(apply_pipeline,
                                    input_columns='text',
                                    batched=True)
    ds_causal.push_to_hub(f'{dataset_name}-{model_name}', private=True)
    return ds_causal


def get_pipeline(
        model_name: str = format_name(MODEL),
        trainer: Optional[Trainer] = None,
        ) -> TextClassificationPipeline:
    import torch

    if trainer is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = trainer.model
        tokenizer = trainer.tokenizer
    device = 0 if torch.cuda.is_available() else 'cpu'
    return TextClassificationPipeline(model=model,
                                      tokenizer=tokenizer,
                                      device=device)


def train_and_apply(
        force_train: bool = FORCE_TRAIN,
        dataset_name: str = DATASET_NAME,
        model_name: str = format_name(MODEL),
        ) -> Tuple[Dataset, Optional[Dict[str, float]]]:
    from gca.data import get_data
    from gca.utils import huggingface_login

    huggingface_login()

    dataset_name = f'{HUGGINGFACE_ACCOUNT}/{dataset_name}'
    dataset = get_data(dataset_name)

    trainer, metrics = None, None
    if force_train:
        trainer, metrics = train()
    pipeline = get_pipeline(model_name, trainer=trainer)

    ds_causal = apply(dataset, pipeline, dataset_name, model_name)
    return ds_causal, metrics


if __name__ == '__main__':
    _, _ = train_and_apply()
