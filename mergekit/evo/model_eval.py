import re
import torch
import re
import io
import math
import sys
import traceback
import numpy as np

from contextlib import redirect_stdout
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    except Exception as e:
        _, _, tb = sys.exc_info()

        last_tb = traceback.extract_tb(tb)[-1]

        line_number = last_tb.lineno

        error_message = f"{type(e).__name__} on line {line_number}: {str(e)}"
        is_executable = False
        difference = np.inf

    return {
        "is_executable": is_executable,
        "difference": difference,
        "model_output": captured_output,
        "error": error_message
    }


def get_data():
    problems = [
        "The coordinates of a parallelogram are (5, 3), (6, 8), (7, 4) and (x, y) and x > 7. What is the valueof x + y",
        "The coordinates of a parallelogram are (5, 3), (6, 8), (7, 4) and (x, y) and x > 7. What is the valueof x + y"
    ]
    answers = [
        "16",
        "16"
    ]

    return problems, answers

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
          unimported packages, or formatting errors (SyntaxError, TypeError). In the last step of the code, print the final
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

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)

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
            print("No code block found.")

    final_accuracy = total_score / len(problems) if problems else 0
    return {
        'reasoning_python': {
            'score': total_score,
            'accuracy': final_accuracy
        }
    }