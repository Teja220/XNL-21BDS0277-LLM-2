from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("squad_v2")

# Reduce dataset to **2%** for **ultra-fast training**
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * 0.02)))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(int(len(dataset["validation"]) * 0.02)))

def preprocess_function(examples):
    inputs = [q + " " + c for q, c in zip(examples["question"], examples["context"])]

    # Reduce sequence length to **128** for fastest processing
    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # Tokenizing answers
    answers = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in examples["answers"]]
    labels = tokenizer(
        text_target=answers,
        padding="max_length",
        truncation=True,
        max_length=128  # Match input max length
    )

    # Set padding tokens to -100 so that loss ignores padding
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels]
        for labels in model_inputs["labels"]
    ]

    return model_inputs

if __name__ == "__main__":
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.save_to_disk("datasets/tokenized/tokenized_squad_v2_gptneo2.7B")
    print("✅ Ultra-fast dataset tokenized and saved.")
