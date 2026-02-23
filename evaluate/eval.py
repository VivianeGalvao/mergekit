import csv
import os
import datetime
import re
import torch
import re
import io
import math
import sys
import traceback
import transformers
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from contextlib import redirect_stdout
from transformers import AutoModelForCausalLM, AutoTokenizer
from func_timeout import func_timeout, FunctionTimedOut


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

max_score = 10e9
TIME_LIMIT = 10

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


LOG_FILE = "resultados/test_log.csv"
def save_checkpoint_csv(data: dict):
    """
    Salva uma linha de resultado no CSV e FORÇA a escrita no disco imediatamente.
    """
    # Adiciona timestamp para saber quando ocorreu
    data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Verifica se o arquivo já existe para decidir se escreve o cabeçalho
    file_exists = os.path.isfile(LOG_FILE)

    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(data)

            # --- O TRUQUE CONTRA TRAVAMENTO ---
            # Isso força o Python a esvaziar o buffer e o Sistema Operacional a gravar no disco físico.
            f.flush()
            os.fsync(f.fileno())

    except Exception as e:
        print(f"Erro ao salvar no CSV (mas a execução continua): {e}")


def score_math_problem(code_string: str, ground_truth_answer: float, tolerance=1e-6) -> dict:
    """
    Avalia o código em duas dimensões: executabilidade e correção.

    AVISO DE SEGURANÇA: exec() pode executar código arbitrário e malicioso.
    Para um sistema real, isso DEVE ser executado em um ambiente sandboxed
    (ex: um container Docker com permissões restritas e timeout).
    """

    is_executable = False
    difference = ground_truth_answer
    captured_output = ""
    error_message = None

    f = io.StringIO()
    try:
        with redirect_stdout(f):
            exec(code_string, {}, {})

        is_executable = True
        captured_output = f.getvalue().strip()

        model_answer = extract_last_number(captured_output)

        if model_answer is not None:
            difference = abs(ground_truth_answer - model_answer)
            # save_checkpoint_csv({'score': model_answer})
        else:
            #try:
            #    max_score = pd.read_csv(LOG_FILE)['score'].max()
            #except FileNotFoundError as e:
            #    max_score = 0
            difference = abs(ground_truth_answer - max_score) + 1

    except Exception as e:
        _, _, tb = sys.exc_info()

        last_tb = traceback.extract_tb(tb)[-1]

        line_number = last_tb.lineno

        error_message = f"{type(e).__name__} on line {line_number}: {str(e)}"
        is_executable = False
        #try:
        #    max_score = pd.read_csv(LOG_FILE)['score'].max()
        #except FileNotFoundError as e:
        #    max_score = 0
        difference = abs(ground_truth_answer - max_score) + 1

    return {
        "is_executable": is_executable,
        "difference": difference,
        "model_output": captured_output,
        "error": error_message
    }


def get_data(partition='train'):

    if partition == 'train':
        df = pd.read_csv("mergekit/data/gsm8k_validation.csv").head(128)
    else:
        df = pd.read_csv("mergekit/data/gsm8k_test.csv")

    return df['problem'].tolist(), df['final_answer'].tolist()


def get_prompt(input: str) -> str:
    prompt = f"""
        You are a powerful agent with broad math knowledge and great python programming skills.
        You need to use python interpreter to do accurate calculation on math equations.
        !!! Remember:
        0. Do not solve the problem via natural language.
        1. USE CODE to solve the problem step by step. The solution should include <code><end_of_code> block.
        2. All calculations should be done in python code. Provide concise reasoning and thinking in the comments of
        the code.
        3. The most related python packages include ‘math‘, ‘sympy‘, ‘scipy‘, and ‘numpy‘.
        4. Please use the following template:
          Question: the input question
          <code>Construct the code step by step. Use <end_of_step> to indicate the end of each step.
          Ensure your code can execute correctly(excluding <end_of_step>) and print the answer. Avoid undefined variables (NameError),
          unimported packages, or formatting errors (SyntaxError, TypeError).
          Avoid using while loops unless absolutely necessary.
          Prefer for loops with clear ranges to prevent infinite execution.
          Explain every variable in each step. In the last step of the code, print the final
          answer. Now! It’s your turn.

        The following is a demonstration example:
        Question: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how
        many times must Terrell lift them in order to lift the same total weight?
        <code>
        # Step 1: Calculate the total weight lifted with two 20-pound weights
        total_weight_20 = 2 * 20 * 12
        <end_of_step>
        # Step 2: Calculate the weight lifted per repetition with two 15-pound weights
        weight_per_rep_15 = 2 * 15
        <end_of_step>
        # Step 3: Calculate the number of repetitions needed to lift the same total weight with two 15-pound weights
        reps_needed = total_weight_20 / weight_per_rep_15
        <end_of_step>
        # Now print the final answer
        print(reps_needed)
        <end_of_code>

        Question: {input}
        <code>
    """
    return prompt

def extract_code(code: str) -> str:

    match = re.search(r"<code>(.*?)<end_of_code>", code, re.DOTALL)

    if match:
        extracted_code = match.group(1)

        cleaned_code = extracted_code.replace("<end_of_step>", "")

        final_code = cleaned_code.strip()

        return final_code
    else:
        return None

def extract_last_number(output_str: str) -> float | None:
    """
    Encontra o último número (inteiro ou float) na string de saída.
    Isso ajuda a ignorar texto extra que o modelo possa ter printado.
    """

    matches = re.findall(r"[-+]?\d*\.\d+|\d+", output_str)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

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
        batch_prompts_text = []
        for problem in batch_problems:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_prompt(problem)}
            ]

            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            batch_prompts_text.append(prompt_text)

        # 3. Tokenização
        model_inputs = tokenizer(
            batch_prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        input_length = model_inputs.input_ids.shape[1]

        # 4. Geração
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
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


def evaluate_math_model(model_id: str, partition: str='train') -> dict:

    problems, answers = get_data(partition)
    print('Tamanho da base: ', len(problems))

    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct',
        padding_side="left",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()

    decoded_texts = generate_in_batches(
        problems,
        model,
        tokenizer,
        batch_size=660
    )

    total_score = 0
    total_execution = 0
    total_hits = 0
    for i, (full_text, true_answer) in enumerate(zip(decoded_texts, answers)):

        code_snippet = extract_code(full_text)

        if code_snippet:
            try:
                score = func_timeout(TIME_LIMIT, score_math_problem, args=(code_snippet, float(true_answer)))
                total_score += score['difference']
                total_execution += int(score['is_executable'])
                if score['difference'] < 10e-6:
                    total_hits += 1

            except FunctionTimedOut:
                print(f"Time limit exceeded ({TIME_LIMIT}s). Applying penalty.")
                total_score += abs(float(true_answer) - max_score) + 1

            except Exception as e:
                print(f"Code execution error: {e}")
                total_score += abs(float(true_answer) - max_score) + 1
        else:
            total_score += abs(float(true_answer) - max_score) + 1
            print("No code block found.")

    print('Score: ', total_score)

    final_accuracy = total_score / len(problems) if problems else 0
    return {
            'score': total_score,
            'total_execution': total_execution,
            'total_hits': total_hits,
            'execution_rate': total_execution / len(problems) if problems else 0,
            'hit_rate': total_hits / len(problems) if problems else 0,
            'accuracy': final_accuracy
        }


def evaluate_qwen_model():
    models = [
        'Qwen/Qwen2.5-3B-Instruct',
        'alpha-ai/qwen2.5-reason-thought-lite',
        'Spestly/Athena-1-3B',
        'prithivMLmods/QwQ-LCoT-3B-Instruct'
    ]

    results = []

    for model in models:
        print("Avaliação do modelo: ", model)
        res = evaluate_math_model(model)
        res['model_name'] = model
        results.append(res)
        print(f'**** Modelo Base - {model} *****')
        print('Score (erro acumulado): ', res['score'])
        print('Total de execuções válidas: ', res['total_execution'])
        print('Total de acertos nas execuções válidas', res['total_hits'])

    return results


def evaluate_merged_model(output_path):
    print("Avaliação do modelo: ", output_path)

    res = evaluate_math_model(output_path)
    res['model_name'] = output_path
    print(f'**** Modelo merged - {output_path} *****')
    print('Score (erro acumulado): ', res['score'])
    print('Total de execuções válidas: ', res['total_execution'])
    print('Total de acertos nas execuções válidas', res['total_hits'])

    return res


def format_row(name, score, exec_val, hits, bold=False):
    # Formata para notação científica básica primeiro: 1.23e+11
    score_str = f"{score:.2e}"

    # Faz a substituição FORA da f-string para evitar o SyntaxError
    # Usamos strings 'raw' (r'...') ou escape duplo (\\) para o LaTeX funcionar
    tex_score = score_str.replace('e+11', r' \times 10^{11}') \
                         .replace('e+12', r' \times 10^{12}') \
                         .replace('e+13', r' \times 10^{13}') \
                         .replace('e+10', r' \times 10^{10}') \
                         .replace('+', '') # Remove o sinal de mais se sobrar

    # Aplica negrito se necessário
    if bold:
        score_tex = f"\\mathbf{{{tex_score}}}"
    else:
        score_tex = f"{tex_score}"

    exec_rate = exec_val * 100
    acc_rate = hits * 100

    row_str = f"{name} & ${score_tex}$ & {exec_rate:.1f}\\% & {acc_rate:.2f}\\% \\\\"
    return row_str

if __name__ == "__main__":

    benchmarks = []
    final_models = []
    best_test_models = []

    set_seed(0)
    transformers.set_seed(0)
    benchmarks = evaluate_qwen_model()

    for seed in range(1, 4):
        set_seed(seed)
        transformers.set_seed(seed)
        final_models.append(
            evaluate_merged_model(
                f'/home/viviane/resultados/cmaes_merged/merge_{seed}/final_model'
            )
        )
        best_test_models.append(
            evaluate_merged_model(
                f'/home/viviane/resultados/cmaes_merged/merge_{seed}/best_test_model'
            )
        )
    print(benchmarks)
    print(final_models)
    print(best_test_models)

    # Imprime o código LaTeX
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Comparativo Detalhado: Benchmarks vs. Modelos Finais vs. Melhores Checkpoints (N=1319)}")
    print(r"\label{tab:comparativo_completo}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Modelo} & \textbf{Score (Erro)} & \textbf{Taxa Exec. Válida (\%)} & \textbf{Acurácia Global (\%)} \\")
    print(r"\midrule")

    print(r"\multicolumn{4}{l}{\textit{Modelos de Referência (Benchmarks)}} \\")
    for d in benchmarks:
        name = d['model_name'].split('/')[-1].replace("_", "\\_")
        print(format_row(name, d['score'], d['execution_rate'], d['hit_rate']))

    print(r"\midrule")
    print(r"\multicolumn{4}{l}{\textit{Experimentos: Final Models (Última Geração)}} \\")
    for i, d in enumerate(final_models):
        name = f"Merge {i+1} (Final)"
        # Verifica se é o melhor score DENTRO do grupo final_models
        is_best = d['score'] == min([x['score'] for x in final_models])
        print(format_row(name, d['score'], d['execution_rate'], d['hit_rate'], bold=is_best))

    # Média Final
    avg_score_final = sum([x['score'] for x in final_models]) / 3
    avg_exec_final = sum([x['execution_rate'] for x in final_models]) / 3
    avg_hits_final = sum([x['hit_rate'] for x in final_models]) / 3

    # Prepara a string da média (fazendo o replace fora também)
    avg_score_str = f"{avg_score_final:.2e}"
    avg_score_tex = avg_score_str.replace('e+11', r' \times 10^{11}').replace('+', '')

    print(r"\textit{\textbf{Média (Final Models)}} & \textit{$" + avg_score_tex + r"$} & \textit{" + f"{(avg_exec_final)*100:.1f}" + r"\%} & \textit{\textbf{" + f"{(avg_hits_final)*100:.2f}" + r"\%}} \\")

    print(r"\midrule")
    print(r"\multicolumn{4}{l}{\textit{Experimentos: Best Test Models (Melhor Validação)}} \\")
    for i, d in enumerate(best_test_models):
        name = f"Merge {i+1} (Best Test)"
        # Verifica se é o melhor score GLOBAL (considerando best e final)
        is_best_overall = d['score'] == min([x['score'] for x in best_test_models + final_models])
        print(format_row(name, d['score'], d['execution_rate'], d['hit_rate'], bold=is_best_overall))

    # Média Best Test
    avg_score_best = sum([x['score'] for x in best_test_models]) / 3
    avg_exec_best = sum([x['execution_rate'] for x in best_test_models]) / 3
    avg_hits_best = sum([x['hit_rate'] for x in best_test_models]) / 3

    # Prepara a string da média best
    avg_score_best_str = f"{avg_score_best:.2e}"
    avg_score_best_tex = avg_score_best_str.replace('e+11', r' \times 10^{11}').replace('+', '')

    print(r"\textit{\textbf{Média (Best Test)}} & \textit{$" + avg_score_best_tex + r"$} & \textit{" + f"{(avg_exec_best)*100:.1f}" + r"\%} & \textit{\textbf{" + f"{(avg_hits_best)*100:.2f}" + r"\%}} \\")

    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\footnotesize{Nota: Best Test refere-se ao checkpoint com menor erro no conjunto de validação durante a evolução.}")
    print(r"\end{table}")
