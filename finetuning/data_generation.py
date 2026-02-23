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

TIME_LIMIT = 10

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data():

    df = pd.read_csv("mergekit/data/gsm8k_validation.csv").iloc[128:]

    return df['problem'].tolist(), df['final_answer'].tolist()


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


def score_math_problem(code_string: str, ground_truth_answer: float, tolerance=1e-6) -> dict:
    """
    Avalia o código em duas dimensões: executabilidade e correção.

    AVISO DE SEGURANÇA: exec() pode executar código arbitrário e malicioso.
    Para um sistema real, isso DEVE ser executado em um ambiente sandboxed
    (ex: um container Docker com permissões restritas e timeout).
    """

    is_executable = False
    difference = None
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

    except Exception as e:
        _, _, tb = sys.exc_info()

        last_tb = traceback.extract_tb(tb)[-1]

        line_number = last_tb.lineno

        error_message = f"{type(e).__name__} on line {line_number}: {str(e)}"
        is_executable = False

    return {
        "is_executable": is_executable,
        "difference": difference,
        "model_output": captured_output,
        "model_answer": model_answer,
        "error": error_message
    }


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

def generate_math_model(model_id: str) -> dict:

    problems, answers = get_data()
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

    dataset_rows = []
    for reasoning, prob, true_answer in zip(decoded_texts, problems, answers):

        code_snippet = extract_code(reasoning)

        exec_output = ""
        is_correct = False

        if code_snippet:
            try:
                res = func_timeout(TIME_LIMIT, score_math_problem, args=(code_snippet, float(true_answer)))
                if res['difference'] < 10e-6:
                    is_correct = True
                exec_output = res['model_output']

            except FunctionTimedOut:
                print(f"Time limit exceeded ({TIME_LIMIT}s). Applying penalty.")

            except Exception as e:
                print(f"Code execution error: {e}")
        else:
            print("No code block found.")

        dataset_rows.append({
            "problem": prob,
            "answer": true_answer,
            "reasoning": reasoning,
            "executed_answer": exec_output,
            "is_correct": is_correct,
            "code_snippet": code_snippet
        })

    return dataset_rows


if __name__ == "__main__":

    models = {
        # "merged_qwen": "/home/viviane/resultados/cmaes_merged/merge_3/final_model",
        "qwen_3b": "Qwen/Qwen2.5-3B-Instruct"
    }

    for model in models:
        generations = generate_math_model(
            model_id=models[model]
        )

        df_result = pd.DataFrame(generations)
        df_result.to_csv(f"mergekit/data/gsm8k_full_generation_log_{model}.csv", index=False)

        print(f"Log completo salvo em 'gsm8k_full_generation_log_{model}.csv' com {len(df_result)} linhas.")
