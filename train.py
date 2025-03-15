import torch
import wandb
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

wandb.init(
    project="gpt-neo-finetune",
    name="fast-training-run",
    config={
        "model_name": "EleutherAI/gpt-neo-2.7B",
        "batch_size": 8,       
        "num_train_epochs": 1, 
        "learning_rate": 2e-4, 
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "logging_steps": 100,
        "eval_steps": 500,     
        "save_steps": 1000     
    }
)

MODEL_NAME = wandb.config.model_name

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with LoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

lora_config = LoraConfig(
    r=wandb.config.lora_r,
    lora_alpha=wandb.config.lora_alpha,
    lora_dropout=wandb.config.lora_dropout,
    target_modules=["k_proj", "v_proj", "q_proj", "out_proj"]
)
model = get_peft_model(model, lora_config)

# Load tokenized dataset
dataset = load_from_disk("datasets/tokenized/tokenized_squad_v2_gptneo2.7B")

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=wandb.config.batch_size,
    per_device_eval_batch_size=wandb.config.batch_size,
    num_train_epochs=wandb.config.num_train_epochs,
    learning_rate=wandb.config.learning_rate,
    logging_steps=wandb.config.logging_steps,
    evaluation_strategy="steps",
    eval_steps=wandb.config.eval_steps,
    save_steps=wandb.config.save_steps,
    logging_dir="./logs",
    report_to="wandb",
    bf16=True,
    deepspeed="ds_config.json",
    label_smoothing_factor=0.1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

def train_model():
    print("🚀 Starting ultra-fast fine-tuning...")
    trainer.train()
    trainer.save_model("models/fine-tuned-gptneo-2.7B")
    print("✅ Training complete in minimal time!")

    # Now evaluate on the validation set
    print("🔍 Evaluating on validation set...")
    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)

    # Optionally compute perplexity if "eval_loss" is provided
    if "eval_loss" in metrics:
        val_loss = metrics["eval_loss"]
        perplexity = (
            round(math.exp(val_loss), 2)
            if val_loss < 20 else float("inf")
        )
        print("Perplexity:", perplexity)

if __name__ == "__main__":
    train_model()
