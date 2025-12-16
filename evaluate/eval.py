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

import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

from contextlib import redirect_stdout
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


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

    df = pd.read_csv("mergekit/data/gsm8k_test.csv").head(75)

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
            # print(code_snippet)
            score = score_math_problem(code_snippet, float(true_answer))
            total_score += score['difference']
        else:
            total_score += abs(float(true_answer) - pd.read_csv(LOG_FILE)['score'].max())
            print("No code block found.")

    print('Score: ', total_score)

    final_accuracy = total_score / len(problems) if problems else 0
    return {
        'reasoning_python': {
            'score': total_score,
            'accuracy': final_accuracy
        }
    }


def evaluate_qwen_model():
    model = 'Qwen/Qwen2.5-3B-Instruct'

    res = evaluate_math_model(model)
    print('**** Modelo Base - Qwen2.5-3B-Instruct *****')
    print(res['reasoning_python']['score'])

    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
            print(f"Arquivo removido: {LOG_FILE}")
        except OSError as e:
            print(f"Erro ao tentar remover o arquivo: {e}")


def evaluate_merged_model(output_path):

    res = evaluate_math_model(output_path)
    print(f'**** Modelo merged - {output_path} *****')
    print(res['reasoning_python']['score'])

    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
            print(f"Arquivo removido: {LOG_FILE}")
        except OSError as e:
            print(f"Erro ao tentar remover o arquivo: {e}")


if __name__ == "__main__":

    configs = [
        '/home/viviane/resultados/cmaes_merged/merge_1/final_model/mergekit_config.yml',
        '/home/viviane/resultados/cmaes_merged/merge_2/final_model/mergekit_config.yml',
        '/home/viviane/resultados/cmaes_merged/merge_3/final_model/mergekit_config.yml'
    ]

    for seed, config in enumerate(configs):
        set_seed(seed+1)
        transformers.set_seed(seed+1)
        evaluate_qwen_model()
        evaluate_merged_model(
            f'/home/viviane/resultados/cmaes_merged/merge_{seed+1}/final_model'
        )
