"""
LLM Fine-Tuning Pipeline (LoRA)
Fine-tunes Llama 3.2 (3B) on domain-specific Q&A using LoRA rank-8 via HuggingFace PEFT + TRL
Reduces trainable parameters by ~99% vs full fine-tune
"""

import os
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig
import evaluate


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

@dataclass
class FineTuneConfig:
    # Model
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: str = "output/lora-llama3-qa"

    # LoRA
    lora_r: int = 8                  # rank-8 as per resume
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True   # as per resume

    # Data
    dataset_name: str = "data/qa_dataset.jsonl"
    val_split: float = 0.1
    seed: int = 42

    # Export
    gguf_output: str = "output/model.gguf"


# ─────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────

def load_qa_dataset(config: FineTuneConfig) -> tuple[Dataset, Dataset]:
    """Load JSONL Q&A dataset and split into train/val."""
    data_path = Path(config.dataset_name)

    if not data_path.exists():
        print(f"[Data] '{data_path}' not found — generating sample dataset...")
        _generate_sample_dataset(data_path)

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = dataset.shuffle(seed=config.seed)

    split = dataset.train_test_split(test_size=config.val_split, seed=config.seed)
    train_ds, val_ds = split["train"], split["test"]

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)} samples")
    return train_ds, val_ds


def _generate_sample_dataset(output_path: Path, n: int = 500):
    """Generate a sample domain Q&A dataset for demonstration."""
    import random
    random.seed(42)

    topics = [
        ("machine learning", [
            ("What is overfitting?", "Overfitting occurs when a model learns training data too well, including noise, leading to poor generalisation on unseen data. It is mitigated through regularisation, dropout, and early stopping."),
            ("Explain gradient descent.", "Gradient descent is an optimisation algorithm that iteratively adjusts model parameters in the direction of the negative gradient of the loss function to minimise it."),
            ("What is a transformer?", "A transformer is a neural network architecture based on self-attention mechanisms. It processes sequences in parallel and forms the foundation of modern LLMs like GPT and Llama."),
        ]),
        ("deep learning", [
            ("What is batch normalisation?", "Batch normalisation normalises layer inputs across a mini-batch, stabilising training and allowing higher learning rates. It reduces internal covariate shift."),
            ("Explain the vanishing gradient problem.", "The vanishing gradient problem occurs in deep networks when gradients become extremely small during backpropagation, preventing early layers from learning effectively."),
            ("What is dropout?", "Dropout is a regularisation technique that randomly deactivates a fraction of neurons during training, preventing co-adaptation and reducing overfitting."),
        ]),
        ("natural language processing", [
            ("What is tokenisation?", "Tokenisation is the process of splitting text into smaller units (tokens) such as words or subwords. Modern LLMs use byte-pair encoding (BPE) for efficient vocabulary coverage."),
            ("What is attention mechanism?", "The attention mechanism allows models to weigh the relevance of different input tokens when producing each output token, enabling long-range dependency modelling."),
            ("What is fine-tuning?", "Fine-tuning adapts a pre-trained model to a specific task by continuing training on a smaller domain-specific dataset, leveraging the general knowledge already encoded in the model."),
        ]),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for _ in range(n):
        domain, qas = random.choice(topics)
        q, a = random.choice(qas)
        # Add slight variation
        records.append({"question": q, "answer": a, "domain": domain})

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[Data] Generated {len(records)} sample Q&A pairs → {output_path}")


def format_prompt(example: dict, tokenizer) -> dict:
    """Format Q&A pair into instruction-following chat format."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer questions accurately and concisely."},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


# ─────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────

def load_base_model(config: FineTuneConfig):
    """Load Llama 3.2 with 4-bit quantisation for memory efficiency."""
    print(f"[Model] Loading {config.base_model} ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def apply_lora(model, config: FineTuneConfig):
    """Wrap model with LoRA adapters (rank-8)."""
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameter stats
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total
    print(f"[LoRA]  Total params   : {total:,}")
    print(f"[LoRA]  Trainable      : {trainable:,}  ({pct:.2f}%)")
    print(f"[LoRA]  Frozen (~99%)  : {total - trainable:,}")

    return model


# ─────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────

def train(config: FineTuneConfig):
    """Full fine-tuning pipeline: load → LoRA → train → eval → export."""

    # 1. Data
    train_ds, val_ds = load_qa_dataset(config)

    # 2. Model
    model, tokenizer = load_base_model(config)
    model = apply_lora(model, config)

    # 3. Format datasets
    train_ds = train_ds.map(lambda x: format_prompt(x, tokenizer))
    val_ds = val_ds.map(lambda x: format_prompt(x, tokenizer))

    # 4. Training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        bf16=False,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
        gradient_checkpointing=config.gradient_checkpointing,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        seed=config.seed,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    print("\n[Train] Starting fine-tuning...\n")
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    print(f"\n[Train] Done in {elapsed/60:.1f} minutes")

    # 6. Save LoRA adapter
    adapter_path = Path(config.output_dir) / "lora-adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[Save]  LoRA adapter → {adapter_path}")

    # 7. Evaluate
    eval_results = evaluate_model(trainer, val_ds, tokenizer, config)

    # 8. Export to GGUF
    export_to_gguf(adapter_path, config)

    return eval_results


# ─────────────────────────────────────────────────────────
# Evaluation (BLEU + ROUGE)
# ─────────────────────────────────────────────────────────

def evaluate_model(trainer, val_ds, tokenizer, config: FineTuneConfig) -> dict:
    """Compute BLEU and ROUGE scores on validation set."""
    print("\n[Eval]  Computing BLEU / ROUGE metrics...")

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    predictions, references = [], []
    model = trainer.model
    model.eval()

    for example in val_ds.select(range(min(50, len(val_ds)))):
        prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predictions.append(generated.strip())
        references.append(example["answer"])

    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references]
    )
    rouge_score = rouge.compute(predictions=predictions, references=references)

    results = {
        "bleu": round(bleu_score["bleu"], 4),
        "rouge1": round(rouge_score["rouge1"], 4),
        "rouge2": round(rouge_score["rouge2"], 4),
        "rougeL": round(rouge_score["rougeL"], 4),
    }

    print(f"[Eval]  BLEU   : {results['bleu']}")
    print(f"[Eval]  ROUGE-1: {results['rouge1']}")
    print(f"[Eval]  ROUGE-2: {results['rouge2']}")
    print(f"[Eval]  ROUGE-L: {results['rougeL']}")

    # Save metrics
    metrics_path = Path(config.output_dir) / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"[Eval]  Saved → {metrics_path}")

    return results


# ─────────────────────────────────────────────────────────
# GGUF Export (for Ollama local inference)
# ─────────────────────────────────────────────────────────

def export_to_gguf(adapter_path: Path, config: FineTuneConfig):
    """
    Merge LoRA weights into base model and export to GGUF for Ollama.
    Requires llama.cpp convert script to be available.
    """
    print("\n[GGUF]  Merging LoRA adapter into base model...")

    merged_path = Path(config.output_dir) / "merged-model"
    merged_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load base + merge adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model = model.merge_and_unload()

        model.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)
        print(f"[GGUF]  Merged model saved → {merged_path}")

        # Convert to GGUF using llama.cpp
        import subprocess
        gguf_out = Path(config.gguf_output)
        gguf_out.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["python3", "scripts/convert_to_gguf.py",
             str(merged_path), "--outfile", str(gguf_out), "--outtype", "q4_k_m"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"[GGUF]  Exported → {gguf_out}")
            print(f"[GGUF]  Run with Ollama: ollama run {gguf_out}")
        else:
            print(f"[GGUF]  llama.cpp not found — run scripts/convert_to_gguf.py manually")
            print(f"        See README for setup instructions.")

    except Exception as e:
        print(f"[GGUF]  Skipped (run on GPU machine): {e}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning Pipeline for Llama 3.2")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", default="data/qa_dataset.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--output", default="output/lora-llama3-qa")
    args = parser.parse_args()

    config = FineTuneConfig(
        base_model=args.model,
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        output_dir=args.output,
    )

    train(config)
