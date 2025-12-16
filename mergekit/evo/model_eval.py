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
import numpy as np
import pandas as pd
import multiprocessing
import sys


from contextlib import redirect_stdout
from contextlib import redirect_stdout
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


LOG_FILE = "resultados/training_log.csv"
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

# # Função auxiliar que roda no processo isolado
# def _worker_exec(code_string, queue):
#     f = io.StringIO()
#     result = {
#         "success": False,
#         "output": "",
#         "error": None
#     }

#     try:
#         with redirect_stdout(f):
#             # Executa o código
#             exec(code_string, {}, {})

#         result["success"] = True
#         result["output"] = f.getvalue().strip()

#     except Exception as e:
#         # Captura o erro e a linha
#         _, _, tb = sys.exc_info()
#         # Tenta pegar o último frame do traceback
#         try:
#             last_tb = traceback.extract_tb(tb)[-1]
#             line_number = last_tb.lineno
#         except:
#             line_number = "?"

#         result["error"] = f"{type(e).__name__} on line {line_number}: {str(e)}"
#         # Mesmo com erro, pegamos o que foi printado antes (pode ser útil)
#         result["output"] = f.getvalue().strip()

#     queue.put(result)

# def score_math_problem(code_string: str, ground_truth_answer: float, timeout: int = 10) -> dict:
#     """
#     Avalia o código com proteção contra Loop Infinito (Timeout).
#     """

#     # 1. Preparação do Multiprocessing
#     queue = multiprocessing.Queue()
#     p = multiprocessing.Process(target=_worker_exec, args=(code_string, queue))

#     p.start()
#     p.join(timeout) # Espera o tempo limite (em segundos)

#     # 2. Verificação do Processo
#     if p.is_alive():
#         # TIMEOUT DETECTADO! Matamos o processo.
#         p.terminate()
#         p.join()

#         is_executable = False
#         captured_output = "" # Ou recuperar o que deu tempo de rodar (difícil com kill)
#         error_message = f"TimeoutError: Execution exceeded {timeout} seconds (Possible Infinite Loop)."
#         worker_result = None
#     else:
#         # Processo terminou a tempo
#         if not queue.empty():
#             worker_result = queue.get()
#             is_executable = worker_result["success"]
#             captured_output = worker_result["output"]
#             error_message = worker_result["error"]
#         else:
#             # Caso raro de crash silencioso
#             is_executable = False
#             captured_output = ""
#             error_message = "Process crashed silently."
#             worker_result = None

#     # 3. Lógica de Negócio (Cálculo da Diferença e Fallback)

#     # Função auxiliar para pegar o max score do CSV sem repetir código
#     def get_fallback_diff():
#         try:
#             if os.path.exists(LOG_FILE):
#                 df = pd.read_csv(LOG_FILE)
#                 if not df.empty and 'score' in df.columns:
#                     max_score = df['score'].max()
#                     # Garante que é um número, senão usa 0
#                     # if pd.isna(max_score): max_score = 0.0
#                     return ground_truth_answer - max_score
#         except Exception:
#             pass
#         return ground_truth_answer # Se não tiver log, penalidade máxima (diferença total)

#     difference = ground_truth_answer # Valor padrão inicial

#     if is_executable:
#         # Tenta extrair a resposta do output
#         model_answer = extract_last_number(captured_output)

#         if model_answer is not None:
#             difference = ground_truth_answer - model_answer
#             # Salva o checkpoint se deu certo (conforme seu código original)
#             save_checkpoint_csv({'score': model_answer})
#             # Nota: Cuidado ao salvar aqui se estiver rodando em paralelo, pode dar conflito de arquivo.
#         else:
#             # Executou mas não printou número
#             difference = get_fallback_diff()
#     else:
#         # Não executou (Erro ou Timeout)
#         difference = get_fallback_diff()

#     return {
#         "is_executable": is_executable,
#         "difference": abs(difference),
#         "model_output": captured_output,
#         "error": error_message
#     }

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
            difference = ground_truth_answer - model_answer
            save_checkpoint_csv({'score': model_answer})
        else:
            max_score = pd.read_csv(LOG_FILE)['score'].max()
            difference = ground_truth_answer - max_score

    except Exception as e:
        _, _, tb = sys.exc_info()

        last_tb = traceback.extract_tb(tb)[-1]

        line_number = last_tb.lineno

        error_message = f"{type(e).__name__} on line {line_number}: {str(e)}"
        is_executable = False
        difference = ground_truth_answer - pd.read_csv(LOG_FILE)['score'].max()

    return {
        "is_executable": is_executable,
        "difference": abs(difference),
        "model_output": captured_output,
        "error": error_message
    }


def get_data():

    df = pd.read_csv("mergekit/data/gsm8k_validation.csv").head(75)

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

def evaluate_math_model(model_id: str) -> dict:
    problems, answers = get_data()

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

    batch_prompts_text = []
    for problem in problems:

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


    model_inputs = tokenizer(
        batch_prompts_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    input_length = model_inputs.input_ids.shape[1]

    print(f"Processando batch de tamanho: {len(problems)}")

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id
        )


    generated_tokens_only = generated_ids[:, input_length:]
    decoded_texts = tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)

    total_score = 0
    for i, (full_text, true_answer) in enumerate(zip(decoded_texts, answers)):

        code_snippet = extract_code(full_text)

        if code_snippet:
            score = score_math_problem(code_snippet, float(true_answer))
            total_score += score['difference']
        else:
            total_score += abs(float(true_answer) - pd.read_csv(LOG_FILE)['score'].max())
            print("No code block found.")

    print('Score: ', total_score)

    final_accuracy = total_score / len(problems) if problems else 0
    return {
        'reasoning_python': {
            'score': -total_score,
            'accuracy': final_accuracy
        }
    }
