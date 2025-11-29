import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re

def extract_final_answer(text: str) -> str:
    """
    No GSM8K, a resposta final vem após '####'.
    Exemplo: '... portanto a resposta é #### 42' -> Retorna '42'
    Removemos vírgulas para facilitar conversão numérica futura (ex: 1,000 -> 1000).
    """
    if "####" not in text:
        return None

    # Pega tudo depois do #### e remove espaços
    answer = text.split("####")[1].strip()

    # Opcional: Remover vírgulas de milhares (ex: 1,200 -> 1200) para manter limpo
    answer = answer.replace(",", "")

    return answer

def process_gsm8k():
    print("1. Carregando o dataset GSM8K oficial...")
    # 'main' é a configuração padrão do dataset
    dataset = load_dataset("openai/gsm8k", "main")

    # O dataset vem dividido em 'train' e 'test' nativamente.
    # Vamos juntar tudo primeiro, conforme seu pedido de "dividir toda a base".
    all_data = {'train': [], 'test': []}

    # Itera sobre os splits originais (train e test) para unificar
    for split in ['train', 'test']:
        for item in dataset[split]:
            problem = item['question']
            raw_answer = item['answer']
            final_answer = extract_final_answer(raw_answer)

            all_data[split].append({
                'problem': problem,
                'final_answer': final_answer,
                'original_split': split # Mantendo rastro de onde veio (opcional)
            })

    # 2. Criando o DataFrame
    df_val = pd.DataFrame(all_data['train'])
    df_test = pd.DataFrame(all_data['test'])
    print(f"Total de linhas carregadas [train]: {len(df_val)}")
    print(f"Total de linhas carregadas [test]: {len(df_test)}")

    # Visualiza as primeiras linhas para garantir que funcionou
    # print("\n--- Amostra dos Dados ---")
    # print(df.head(3))

    # # 3. Dividindo em Validação e Teste
    # # Aqui vou assumir que você quer dividir TUDO em dois sets novos.
    # # test_size=0.2 significa que 20% vai para teste e 80% para validação.
    # # random_state=42 garante que a divisão seja sempre a mesma (reprodutibilidade).
    # df_val, df_test = train_test_split(df, test_size=0.2, random_state=42)

    print(f"\n--- Divisão Concluída ---")
    print(f"Conjunto de Validação: {len(df_val)} exemplos")
    print(f"Conjunto de Teste: {len(df_test)} exemplos")

    return df_val, df_test

# Executando
if __name__ == "__main__":
    df_validation, df_test = process_gsm8k()

    # Opcional: Salvar em CSV/JSONL para usar depois
    df_validation.to_csv("gsm8k_validation.csv", index=False)
    df_test.to_csv("gsm8k_test.csv", index=False)