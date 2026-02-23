import re
import math
import json
import transformers
import numpy as np
import pandas as pd
from tqdm import tqdm

from unsloth import FastLanguageModel
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def extract_last_number(text):
    """
    Procura o último número presente em um texto.
    Lida com formatos como: 1,000 | 18.5 | $20 | -5
    """
    if not isinstance(text, str):
        return None

    # 1. Limpeza básica para evitar confundir o parser
    # Remove vírgulas de milhar (ex: 1,000 -> 1000)
    text_clean = text.replace(',', '')

    # 2. Regex para encontrar números (inteiros e floats)
    # Explicação: Opcional sinal (-) + Digitos + Opcional Ponto Decimal + Digitos
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text_clean)

    if matches:
        # Pega o ÚLTIMO número encontrado no texto (assumindo que a resposta está no final)
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

def calculate_accuracy(predictions_text, ground_truths_numeric):
    correct_count = 0
    total_count = len(predictions_text)

    print(f"{'PRED (Raw)':<30} | {'EXTRAÍDO':<10} | {'GABARITO':<10} | {'STATUS'}")
    print("-" * 65)

    nums = []
    for pred_text, truth_num in zip(predictions_text, ground_truths_numeric):

        # 1. Extração
        extracted_num = extract_last_number(pred_text)
        nums.append(extracted_num)

        # 2. Comparação (Lógica de Tolerância)
        is_correct = False
        if extracted_num is not None:
            if abs(extracted_num - float(truth_num)) < 1e-6:
                is_correct = True

        if is_correct:
            correct_count += 1

        # Log para visualização
        status = "✅" if is_correct else "❌"
        # Corta o texto longo só para exibir na tabela
        display_text = (pred_text[:27] + '...') if len(pred_text) > 30 else pred_text
        print(f"{display_text:<30} | {str(extracted_num):<10} | {str(truth_num):<10} | {status}")

    accuracy = correct_count / total_count
    return accuracy, nums


def get_data(partition='train'):

    if partition == 'train':
        df = pd.read_csv("mergekit/data/gsm8k_validation.csv").head(128)
    else:
        df = pd.read_csv("mergekit/data/gsm8k_test.csv")

    return df['problem'].tolist(), df['final_answer'].tolist()


def generate_in_batches(problems, model, tokenizer, batch_size=16):
    all_results = []

    # Garante que o padding seja feito à esquerda para geração (padrão para decoder-only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Iniciando geração para {len(problems)} problemas com batch_size={batch_size}...")

    # Loop principal: caminha de 'batch_size' em 'batch_size'
    for i in tqdm(range(0, len(problems), batch_size), desc="Processando Batches"):

        # 1. Seleciona a fatia atual dos problemas
        batch_problems = problems[i : i + batch_size]

        # 2. Prepara os prompts APENAS para esse batch
        alpaca_prompt = """
        Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.
        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        """
        instruction_text = "Solve the following math problem step-by-step."
        batch_prompts_text = []
        for problem in batch_problems:
            text = alpaca_prompt.format(
                instruction_text,
                problem
                )
            batch_prompts_text.append(text)

        # 3. Tokenização
        model_inputs = tokenizer(
            batch_prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        input_length = model_inputs.input_ids.shape[1]

        # 4. Geração
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "### Instruction:", "###"]
        stop_token_ids = tokenizer.convert_tokens_to_ids(stop_tokens)
        stop_token_ids = [t for t in stop_token_ids if t is not None]

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                eos_token_id=[tokenizer.eos_token_id] + stop_token_ids,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id
            )

        # 5. Decodificação (Remove o prompt de entrada e converte tokens em texto)
        # Pegamos apenas os tokens novos gerados (do input_length para frente)
        generated_only_ids = generated_ids[:, input_length:]

        decoded_output = tokenizer.batch_decode(
            generated_only_ids,
            skip_special_tokens=True
        )

        # 6. Acumula os resultados
        all_results.extend(decoded_output)

        # Limpeza opcional para garantir memória livre entre batches
        del model_inputs, generated_ids, generated_only_ids
        torch.cuda.empty_cache()

    return all_results


def evaluate_model(model_id: str, partition: str='train') -> dict:

    problems, answers = get_data(partition)
    print('Tamanho da base: ', len(problems))

    # Configurações Fixas
    max_seq_length = 2048
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    decoded_texts = generate_in_batches(
        problems,
        model,
        tokenizer,
        batch_size=660
    )

    accuracy, preds = calculate_accuracy(decoded_texts, answers)
    print(f"✅ Acurácia do modelo '{model_id}' na partição '{partition}': {accuracy*100:.2f}%")

    return {
        'decoded_texts': decoded_texts,
        'answers': answers,
        'preds': preds,
        'accuracy': round(accuracy*100, 2)
    }


def evaluate_qwen_model(test=True):
    models = [
        'unsloth/Qwen2.5-7B-Instruct',
        'unsloth/Qwen2.5-3B-Instruct'
    ]

    results = []

    for model in models:
        print("Avaliação do modelo: ", model)
        res = evaluate_model(model, partition='test' if test else 'train')
        res['model_name'] = model
        results.append(res)
    return results


def evaluate_ft_model(test=True):
    models = [
        '/home/viviane/resultados/finetunig/unsloth_Qwen2_5-3B-Instruct__gsm8k_full_generation_log_merged_qwen',
        '/home/viviane/resultados/finetunig/unsloth_Qwen2_5-3B-Instruct__gsm8k_full_generation_log_qwen_3b'
    ]

    results = []
    for model in models:
        print("Avaliação do modelo: ", model)

        res = evaluate_model(model, partition='test' if test else 'train')
        res['model_name'] = model
        results.append(res)

    return results

if __name__ == "__main__":

    benchmarks = []
    ft_models = []

    set_seed(0)
    transformers.set_seed(0)
    benchmarks = evaluate_qwen_model()
    ft_models = evaluate_ft_model()

    # 1. Cria um dicionário único contendo as duas listas
    dados_finais = {
        "benchmarks_baseline": benchmarks,
        "models_finetuned": ft_models
    }

    # 2. Define o nome do arquivo
    output_file = "resultados_completos_eval_ft.json"

    # 3. Salva no arquivo
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                dados_finais,
                f,
                indent=4,              # Deixa o JSON legível (bonito)
                ensure_ascii=False,    # Garante que acentos funcionem
                cls=NumpyEncoder       # Converte tipos do NumPy automaticamente
            )
        print(f"✅ Resultados salvos com sucesso em '{output_file}'")

    except Exception as e:
        print(f"❌ Erro ao salvar JSON: {e}")

    # Opcional: print para debug
    # print(json.dumps(dados_finais, indent=2, cls=NumpyEncoder))
