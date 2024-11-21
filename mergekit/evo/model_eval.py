import random
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, pipeline, BertForSequenceClassification, AutoConfig
from datasets import load_dataset


def eval_task(pipe, task):

    if task == 'fillmask_sentiment_pt':
        targets=['positivo', 'negativo']
        data_val = load_dataset('csv', data_files='mergekit/data/maritaca-ai_sst2_pt.csv')
        tokenizer_kwargs = {"truncation": True, "max_length":512}
        vals = data_val['train'].map(
            lambda x: pipe(
                x['text_fillmask'],
                top_k=1,
                targets=targets,
                tokenizer_kwargs=tokenizer_kwargs
            )[0]
        )
        df = pd.DataFrame(vals)

        score = df['score'].mean()
        f1 = f1_score(
            df['sentiment'].replace('positivo', 1).replace('negativo', 0), 
            df['token_str'].replace('positivo', 1).replace('negativo', 0), 
            average='binary'
        )

        acc = accuracy_score(
            df['sentiment'].replace('positivo', 1).replace('negativo', 0),
            df['token_str'].replace('positivo', 1).replace('negativo', 0),
        )

        results = {
            'sentiment_pt': {
                'score': score, 
                'f1-score': f1,
                'accuracy': acc,
                'alias': 'sst2_pt'
            }
        }

        return results
    
    if task == 'sentiment_pt':

        model_name = "danielribeiro/google-play-sentiment-analysis"

        model = BertForSequenceClassification.from_pretrained(model_name)

        # freeze classification layer from base model
        pipe.model.classifier = model.classifier.to('cuda')
        pipe.model.config.id2label = model.config.id2label

        tokenizer_kwargs = {
            'padding':True,
            'truncation':True,
            'max_length':512
        }

        data_val = load_dataset('csv', data_files='mergekit/data/maritaca-ai_sst2_pt.csv')
        vals = data_val['train'].map(
            lambda x: pipe(x['text'], **tokenizer_kwargs)[0]
        )
        df = pd.DataFrame(vals)
        df['model_label'] = df['label'].replace('Positivo', 1).replace('Negativo', 0).replace('Neutro', -1)

        f1 = f1_score(
            df[df['label']!='Neutro']['true_label'],
            df[df['label']!='Neutro']['model_label'],
            average='binary'
        )

        acc = accuracy_score(
            df[df['label']!='Neutro']['true_label'],
            df[df['label']!='Neutro']['model_label'],
        )

        results = {
            'sentiment_pt': {
                'score': f1*100,
                'f1-score': f1,
                'accuracy': acc,
            }
        }

        return results

    if task == 'hatebr':

        model_name = "Silly-Machine/TuPy-Bert-Large-Binary-Classifier"

        model = BertForSequenceClassification.from_pretrained(model_name)

        # freeze classification layer from base model
        pipe.model.classifier = model.classifier.to('cuda')
        pipe.model.config.id2label = model.config.id2label

        tokenizer_kwargs = {
            'padding':True,
            'truncation':True,
            'max_length':512
        }

        data_val = load_dataset('csv', data_files='mergekit/data/hatebr.csv')
        vals = data_val['train'].map(
            lambda x: pipe(x['text'], **tokenizer_kwargs)[0]
        )

        df = pd.DataFrame(vals)
        df['model_label'] = df['label'].replace('hate', True).replace('not hate', False)

        f1 = f1_score(
            df['true_label'],
            df['model_label'],
            average='binary'
        )

        acc = accuracy_score(
            df['true_label'],
            df['model_label'],
        )

        results = {
            'hatebr': {
                'score': f1*100,
                'f1-score': f1,
                'accuracy': acc,
            }
        }

        return results


def fillmask_evaluator(
        merged_path,
        task
):
    
    # getting the BertMLM layer from base model
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)
    pipe = pipeline(
            task="fill-mask",
            model='neuralmind/bert-large-portuguese-cased',
            tokenizer=tokenizer,
            device='cuda'
        )

    tokenizer = AutoTokenizer.from_pretrained(
        merged_path,
        do_lower_case=False,
    )

    fill_mask = pipeline(
        task="fill-mask",
        model=merged_path,
        tokenizer=tokenizer,
        device='cuda'
    )
    fill_mask.model.cls = pipe.model.cls

    del pipe

    return eval_task(fill_mask, task)

def classification_evaluator(
        merged_path,
        task
):
    
    print(f'Avaliando modelo {merged_path}')
    
    pipe = pipeline(
        "text-classification", 
        model=merged_path,
        tokenizer=merged_path,
        device='cuda',
        truncation=True
    )

    return eval_task(pipe, task)
    