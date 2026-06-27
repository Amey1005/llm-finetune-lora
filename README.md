# 🦙 LLM Fine-Tuning Pipeline (LoRA)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-PEFT-yellow)](https://huggingface.co/docs/peft)
[![TRL](https://img.shields.io/badge/TRL-SFTTrainer-orange)](https://huggingface.co/docs/trl)
[![LoRA](https://img.shields.io/badge/LoRA-rank--8-purple)](https://arxiv.org/abs/2106.09685)

Fine-tunes **Llama 3.2 (3B)** on a domain-specific Q&A dataset using **LoRA rank-8** via HuggingFace PEFT and TRL. Reduces trainable parameters by **~99%** while achieving competitive task accuracy vs full fine-tuning.

---

## 🏗️ Pipeline Overview

```
Raw Q&A JSONL
      │
      ▼
┌─────────────────────┐
│  Preprocessing      │  validate → clean → tokenizer alignment → split
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  LoRA Fine-Tuning   │  Llama 3.2-3B + rank-8 LoRA + gradient checkpointing
│  (PEFT + TRL)       │  ~99% params frozen, <1% trainable
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Evaluation         │  BLEU + ROUGE vs base model baseline
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  GGUF Export        │  Merge adapter → GGUF (Q4_K_M) → Ollama Modelfile
└─────────────────────┘
```

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Base model** | `meta-llama/Llama-3.2-3B-Instruct` |
| **LoRA rank** | 8 (alpha=16) |
| **Trainable params** | ~0.64% of total (~20M / 3B) |
| **Quantisation** | 4-bit NF4 (QLoRA) via bitsandbytes |
| **Gradient checkpointing** | ✅ Enabled (reduces GPU memory ~40%) |
| **Eval metrics** | BLEU + ROUGE-1/2/L |
| **Export** | GGUF Q4_K_M for Ollama local inference |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
git clone https://github.com/Amey1005/llm-finetune-lora.git
cd llm-finetune-lora
pip install -r requirements.txt
```

### 2. Set HuggingFace token
```bash
cp .env.example .env
# Add your HF_TOKEN (accept Llama 3.2 license on HuggingFace first)
```

### 3. Preprocess dataset
```bash
python src/preprocess.py --input data/qa_dataset.jsonl --output data/qa_clean.jsonl
```

### 4. Fine-tune
```bash
python src/finetune.py --dataset data/qa_clean.jsonl --epochs 3 --lora-r 8
```

### 5. Evaluate
```bash
python src/evaluate.py --adapter output/lora-llama3-qa/lora-adapter --test-data data/splits/test.jsonl
```

### 6. Export to GGUF + run with Ollama
```bash
python scripts/export_gguf.py --adapter output/lora-llama3-qa/lora-adapter
ollama create llama3-qa -f output/Modelfile
ollama run llama3-qa
```

---

## 📊 Parameter Efficiency (LoRA rank-8)

```
Total parameters     : 3,212,749,824  (3.2B)
Trainable (LoRA)     :    20,561,920  (0.64%)
Frozen               : 3,192,187,904  (99.36%)

vs Full Fine-tuning  : 3.2B trainable params
Reduction            : ~99.4% fewer trainable params
```

---

## 📁 Project Structure

```
llm-finetune-lora/
├── src/
│   ├── finetune.py       # LoRA training pipeline (main)
│   ├── preprocess.py     # Data cleaning + tokenizer alignment
│   └── evaluate.py       # BLEU/ROUGE evaluation
├── scripts/
│   └── export_gguf.py    # Merge + GGUF export for Ollama
├── data/                 # Your Q&A JSONL dataset
├── output/               # Checkpoints, adapter, GGUF model
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠️ Tech Stack

- **HuggingFace PEFT** — LoRA adapter training
- **TRL (SFTTrainer)** — Supervised fine-tuning
- **bitsandbytes** — 4-bit QLoRA quantisation
- **evaluate** — BLEU + ROUGE metrics
- **llama.cpp** — GGUF conversion
- **Ollama** — Local model serving

---

*Built by [Amey Kushare](https://github.com/Amey1005)*
