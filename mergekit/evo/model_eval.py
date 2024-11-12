import random
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, pipeline 
from datasets import load_dataset


def eval_task(pipe, task):

    if task == 'sentiment_pt':
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

        score = df['score'].mean()*1000
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


def fillmask_evaluator(
        merged_path,
        task
):

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
    print(fill_mask.tokenizer)

    return eval_task(fill_mask, task)
