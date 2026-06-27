"""
Evaluation Script — BLEU & ROUGE metrics
Compares fine-tuned model vs base model on the test set
"""

import json
import argparse
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import evaluate


def load_model(base_model: str, adapter_path: str = None):
    """Load base model, optionally with LoRA adapter merged."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path:
        print(f"[Eval] Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_eval(
    base_model: str,
    adapter_path: str,
    test_data: str,
    n_samples: int = 100,
    output_path: str = "output/eval_results.json",
):
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    dataset = load_dataset("json", data_files=test_data, split="train")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    results = {}

    for label, adapter in [("base", None), ("finetuned", adapter_path)]:
        print(f"\n[Eval] Evaluating {label} model ({len(dataset)} samples)...")
        model, tokenizer = load_model(base_model, adapter)

        preds, refs = [], []
        for i, ex in enumerate(dataset):
            pred = generate_answer(model, tokenizer, ex["question"])
            preds.append(pred)
            refs.append(ex["answer"])
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(dataset)}]")

        bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
        rouge = rouge_metric.compute(predictions=preds, references=refs)

        results[label] = {
            "bleu": round(bleu["bleu"], 4),
            "rouge1": round(rouge["rouge1"], 4),
            "rouge2": round(rouge["rouge2"], 4),
            "rougeL": round(rouge["rougeL"], 4),
        }

        del model
        torch.cuda.empty_cache()

    # Print comparison
    print(f"\n{'='*55}")
    print(f"{'Metric':<15} {'Base Model':>15} {'Fine-tuned':>15} {'Δ':>8}")
    print(f"{'='*55}")
    for metric in ["bleu", "rouge1", "rouge2", "rougeL"]:
        base_val = results["base"][metric]
        ft_val = results["finetuned"][metric]
        delta = ft_val - base_val
        print(f"{metric:<15} {base_val:>15.4f} {ft_val:>15.4f} {delta:>+8.4f}")
    print(f"{'='*55}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"\n[Eval] Results saved → {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter", default="output/lora-llama3-qa/lora-adapter")
    parser.add_argument("--test-data", default="data/splits/test.jsonl")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", default="output/eval_results.json")
    args = parser.parse_args()

    run_eval(args.base_model, args.adapter, args.test_data, args.n_samples, args.output)
