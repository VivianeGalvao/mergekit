import emoji
import pandas as pd
from datasets import load_dataset


def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')


def prep_hatebr(col):

    ds = load_dataset("ruanchaves/hatebr")
    ds = ds[col].rename_column('offensive_language', 'true_label')
    df = pd.DataFrame(data=ds)
    # filter only offensive and hate speech
    df = df[df['offensive_&_non-hate_speech'] == False]
    df['text'] = df['instagram_comments'].apply(
        lambda x: remove_emojis(x)
    ).str.replace('\s+', ' ', regex=True)
    df.to_csv('data/hatebr.csv', index=False)


def prep_sst2(col):

    ds = load_dataset("maritaca-ai/sst2_pt")
    
    data_val = ds[col].rename_column('label', 'true_label')
    df_sst2 = pd.DataFrame(data=data_val)
    df_sst2['text_fillmask'] = df_sst2['text'] + '. sentimento [MASK]'
    df_sst2['sentiment'] = df_sst2['true_label']\
        .replace(1, 'positivo').replace(0, 'negativo').replace(-1, 'neutro')
    df_sst2.to_csv('data/maritaca-ai_sst2_pt.csv', index=False)


def prep_imdb(col):

    ds = load_dataset("maritaca-ai/imdb_pt")
    imdb_val = ds[col].rename_column('label', 'true_label')
    df_imdb = pd.DataFrame(data=imdb_val)
    df_imdb['text_fillmask'] = df_imdb['text'] + '. sentimento [MASK]'
    df_imdb['sentiment'] = df_imdb['true_label']\
        .replace(1, 'positivo').replace(0, 'negativo').replace(-1, 'neutro')
    df_imdb.to_csv('data/maritaca-ai_imdb_pt.csv', index=False)


def prep_sentiment_base(col):

    ds = load_dataset("maritaca-ai/sst2_pt")    
    data_val = ds[col].rename_column('label', 'true_label')
    df_sst2 = pd.DataFrame(data=data_val)
    df_sst2['text_fillmask'] = df_sst2['text'] + '. sentimento [MASK]'
    df_sst2['sentiment'] = df_sst2['true_label']\
        .replace(1, 'positivo').replace(0, 'negativo').replace(-1, 'neutro')
    
    ds = load_dataset("maritaca-ai/imdb_pt")
    imdb_val = ds[col].rename_column('label', 'true_label')
    df_imdb = pd.DataFrame(data=imdb_val)
    df_imdb['text_fillmask'] = df_imdb['text'] + '. sentimento [MASK]'
    df_imdb['sentiment'] = df_imdb['true_label']\
        .replace(1, 'positivo').replace(0, 'negativo').replace(-1, 'neutro')    

    df = pd.concat([df_sst2, df_imdb], axis=0)

    df.to_csv('data/maritaca-ai_sst2_imdb_pt.csv', index=False)



def main():
    col='validation'
    prep_sst2(col)
    prep_hatebr(col)

main()
