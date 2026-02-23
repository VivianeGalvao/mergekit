import pandas as pd

# Data provided by the user
data = [
    {"score": 1360008667589.0376, "total_execution": 1187, "total_hits": 971, "model_name": "Qwen/Qwen2.5-3B-Instruct"},
    {"score": 1280003012128.5981, "total_execution": 1192, "total_hits": 957, "model_name": "alpha-ai/qwen2.5-reason-thought-lite"},
    {"score": 11609991388101.209, "total_execution": 158, "total_hits": 99, "model_name": "Spestly/Athena-1-3B"},
    {"score": 13179990992231.0, "total_execution": 1, "total_hits": 1, "model_name": "prithivMLmods/QwQ-LCoT-3B-Instruct"},
    {"score": 381641353612.3142, "total_execution": 1281, "total_hits": 953, "model_name": "/home/viviane/resultados/cmaes_merged/merge_1/final_model"},
    {"score": 611640935620.5701, "total_execution": 1258, "total_hits": 965, "model_name": "/home/viviane/resultados/cmaes_merged/merge_2/final_model"},
    {"score": 380007234545.19086, "total_execution": 1281, "total_hits": 1044, "model_name": "/home/viviane/resultados/cmaes_merged/merge_3/final_model"},
    {"score": 431641723813.81665, "total_execution": 1277, "total_hits": 1019, "model_name": "/home/viviane/resultados/cmaes_merged/merge_1/best_test_model"},
    {"score": 370008776485.4369, "total_execution": 1283, "total_hits": 1051, "model_name": "/home/viviane/resultados/cmaes_merged/merge_2/best_test_model"},
    {"score": 400006871044.0189, "total_execution": 1280, "total_hits": 1048, "model_name": "/home/viviane/resultados/cmaes_merged/merge_3/best_test_model"}
]

TOTAL_SAMPLES = pd.read_csv('mergekit/data/gsm8k_test.csv').shape[0]
print(f"Total samples: {TOTAL_SAMPLES}")

# Organizing data
benchmarks = data[:4]
final_models = data[4:7]
best_test_models = data[7:]

def format_row(name, score, exec_val, hits, bold=False):
    # Formata para notação científica básica primeiro: 1.23e+11
    score_str = f"{score:.2e}"

    # Faz a substituição FORA da f-string para evitar o SyntaxError
    # Usamos strings 'raw' (r'...') ou escape duplo (\\) para o LaTeX funcionar
    tex_score = score_str.replace('e+11', r' \times 10^{11}') \
                         .replace('e+12', r' \times 10^{12}') \
                         .replace('e+13', r' \times 10^{13}') \
                         .replace('+', '') # Remove o sinal de mais se sobrar

    # Aplica negrito se necessário
    if bold:
        score_tex = f"\\mathbf{{{tex_score}}}"
    else:
        score_tex = f"{tex_score}"

    exec_rate = (exec_val / TOTAL_SAMPLES) * 100
    acc_rate = (hits / TOTAL_SAMPLES) * 100

    row_str = f"{name} & ${score_tex}$ & {exec_rate:.1f}\\% & {acc_rate:.2f}\\% \\\\"
    return row_str

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
    print(format_row(name, d['score'], d['total_execution'], d['total_hits']))

print(r"\midrule")
print(r"\multicolumn{4}{l}{\textit{Experimentos: Final Models (Última Geração)}} \\")
for i, d in enumerate(final_models):
    name = f"Merge {i+1} (Final)"
    # Verifica se é o melhor score DENTRO do grupo final_models
    is_best = d['score'] == min([x['score'] for x in final_models])
    print(format_row(name, d['score'], d['total_execution'], d['total_hits'], bold=is_best))

# Média Final
avg_score_final = sum([x['score'] for x in final_models]) / 3
avg_exec_final = sum([x['total_execution'] for x in final_models]) / 3
avg_hits_final = sum([x['total_hits'] for x in final_models]) / 3

# Prepara a string da média (fazendo o replace fora também)
avg_score_str = f"{avg_score_final:.2e}"
avg_score_tex = avg_score_str.replace('e+11', r' \times 10^{11}').replace('+', '')

print(r"\textit{\textbf{Média (Final Models)}} & \textit{$" + avg_score_tex + r"$} & \textit{" + f"{(avg_exec_final/TOTAL_SAMPLES)*100:.1f}" + r"\%} & \textit{\textbf{" + f"{(avg_hits_final/TOTAL_SAMPLES)*100:.2f}" + r"\%}} \\")

print(r"\midrule")
print(r"\multicolumn{4}{l}{\textit{Experimentos: Best Test Models (Melhor Validação)}} \\")
for i, d in enumerate(best_test_models):
    name = f"Merge {i+1} (Best Test)"
    # Verifica se é o melhor score GLOBAL (considerando best e final)
    is_best_overall = d['score'] == min([x['score'] for x in best_test_models + final_models])
    print(format_row(name, d['score'], d['total_execution'], d['total_hits'], bold=is_best_overall))

# Média Best Test
avg_score_best = sum([x['score'] for x in best_test_models]) / 3
avg_exec_best = sum([x['total_execution'] for x in best_test_models]) / 3
avg_hits_best = sum([x['total_hits'] for x in best_test_models]) / 3

# Prepara a string da média best
avg_score_best_str = f"{avg_score_best:.2e}"
avg_score_best_tex = avg_score_best_str.replace('e+11', r' \times 10^{11}').replace('+', '')

print(r"\textit{\textbf{Média (Best Test)}} & \textit{$" + avg_score_best_tex + r"$} & \textit{" + f"{(avg_exec_best/TOTAL_SAMPLES)*100:.1f}" + r"\%} & \textit{\textbf{" + f"{(avg_hits_best/TOTAL_SAMPLES)*100:.2f}" + r"\%}} \\")

print(r"\bottomrule")
print(r"\end{tabular}%")
print(r"}")
print(r"\footnotesize{Nota: Best Test refere-se ao checkpoint com menor erro no conjunto de validação durante a evolução.}")
print(r"\end{table}")