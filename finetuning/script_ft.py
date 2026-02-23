import argparse
import os
import torch

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer



def main():
    # 1. Configurar o Parser de Argumentos
    parser = argparse.ArgumentParser(description="Script de Fine-Tuning com Unsloth")

    # Definir os argumentos (flags)
    parser.add_argument("--model_name", type=str, required=True,
                        help="Nome ou caminho do modelo (ex: unsloth/Qwen2.5-7B)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Caminho para o arquivo CSV do dataset")
    parser.add_argument("--output_dir", type=str, default="resultados/finetunig",
                        help="Pasta base para salvar os resultados")

    # Ler os argumentos
    args = parser.parse_args()

    # Variáveis recebidas do terminal
    model_name = args.model_name
    dataset_csv_path = args.dataset_name

    print(f"🔄 Iniciando Fine-Tuning...")
    print(f"   Modelo: {model_name}")
    print(f"   Dataset: {dataset_csv_path}")

    # Configurações Fixas
    max_seq_length = 2048
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_in_4bit = False

    # 2. Carregar Modelo (Usando a variável model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instruction_text = "Solve the following math problem step-by-step."
        inputs      = examples["problem"]
        reasonings  = examples["reasoning"]
        answers     = examples["answer"]
        texts = []
        for input_text, reasoning, answer in zip(inputs, reasonings, answers):
            output_text = f"{reasoning} The final answer is: {str(answer)}"
            text = alpaca_prompt.format(instruction_text, input_text, output_text) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    # 3. Carregar Dataset (Usando a variável dataset_csv_path)
    dataset = load_dataset('csv', data_files=dataset_csv_path)['train']
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        packing = False, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8,  # Aumentado: Batch efetivo 2*8 = 16 (mais estável)
            warmup_steps = 10,                # Um pouco mais de aquecimento evita choques iniciais
            num_train_epochs = 1,             # Reduzido: Evita decorar os dados (Overfitting)
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,              # Aumentado levemente para regularização
            lr_scheduler_type = "cosine",     # "cosine" costuma dar um resultado final melhor que "linear"
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
        ),
    )

    trainer_stats = trainer.train()

    # 4. Salvamento Inteligente (Sanitização de nomes)
    # Remove caracteres especiais para criar nomes de pasta válidos
    safe_model_name = model_name.replace("/", "_").replace(".", "_")
    safe_dataset_name = os.path.basename(dataset_csv_path).replace(".csv", "")

    final_save_path = os.path.join(args.output_dir, f"{safe_model_name}__{safe_dataset_name}")

    print(f"💾 Salvando modelo final em: {final_save_path}")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("Training completed.")

if __name__ == "__main__":
    main()