import re
import torch
import re
import io
import sys
import traceback
import json

import torch

from contextlib import redirect_stdout
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
        else:
            difference = abs(ground_truth_answer - max_score) + 1

    except Exception as e:
        _, _, tb = sys.exc_info()

        last_tb = traceback.extract_tb(tb)[-1]

        line_number = last_tb.lineno

        error_message = f"{type(e).__name__} on line {line_number}: {str(e)}"
        is_executable = False
        difference = abs(ground_truth_answer - max_score) + 1

    return {
        "is_executable": is_executable,
        "difference": difference,
        "model_output": captured_output,
        "error": error_message
    }


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


def evaluate_math_model(decoded_texts: list[str], answers: list[str]) -> dict:

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

    final_accuracy = total_score / len(decoded_texts) if decoded_texts else 0
    return {
            'score': total_score,
            'total_execution': total_execution,
            'total_hits': total_hits,
            'execution_rate': total_execution / len(decoded_texts) if decoded_texts else 0,
            'hit_rate': total_hits / len(decoded_texts) if decoded_texts else 0,
            'accuracy': final_accuracy
        }


if __name__ == "__main__":

    with open("resultados_completos_eval_ft.json", 'r') as f:
        ft_results = json.load(f)['models_finetuned']

    for i, result in enumerate(ft_results):
        print(f"--- Avaliando Modelo Fine-Tuned {result['model_name']} ---")
        decoded_texts = result['decoded_texts']
        answers = result['answers']

        eval_metrics = evaluate_math_model(decoded_texts, answers)

        print(f"Score Total: {eval_metrics['score']:.2e}")
        print(f"Taxa de Execução Válida: {eval_metrics['execution_rate']*100:.2f}%")
        print(f"Acurácia Global: {eval_metrics['hit_rate']*100:.2f}%")
        print("\n")
