"""
Export fine-tuned LoRA model to GGUF format for Ollama local inference.

Steps:
1. Merge LoRA adapter into base model weights
2. Save merged model in HuggingFace format
3. Convert to GGUF using llama.cpp
4. Create Ollama Modelfile

Usage:
    python scripts/export_gguf.py --adapter output/lora-llama3-qa/lora-adapter
"""

import os
import argparse
import subprocess
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter(base_model: str, adapter_path: str, output_dir: str):
    """Merge LoRA weights into base model."""
    print(f"[Merge] Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu"
    )

    print(f"[Merge] Applying LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("[Merge] Adapter merged successfully")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"[Merge] Saved merged model → {out}")
    return out


def convert_to_gguf(merged_dir: Path, output_file: str, quant_type: str = "q4_k_m"):
    """Convert merged HuggingFace model to GGUF using llama.cpp."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if llama.cpp convert script is available
    convert_script = Path("scripts/llama_cpp/convert_hf_to_gguf.py")
    if not convert_script.exists():
        print(f"""
[GGUF] llama.cpp convert script not found at {convert_script}

To set up llama.cpp conversion:
  git clone https://github.com/ggerganov/llama.cpp scripts/llama_cpp
  pip install -r scripts/llama_cpp/requirements.txt

Then re-run this script.
""")
        # Write a placeholder so the repo shows the workflow
        output_file.write_text("# GGUF model placeholder - run export_gguf.py with llama.cpp set up\n")
        return

    print(f"[GGUF]  Converting to GGUF ({quant_type})...")
    result = subprocess.run(
        ["python3", str(convert_script), str(merged_dir),
         "--outfile", str(output_file),
         "--outtype", quant_type],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        size_mb = output_file.stat().st_size / 1e6
        print(f"[GGUF]  Exported → {output_file}  ({size_mb:.0f} MB)")
    else:
        print(f"[GGUF]  Conversion failed:\n{result.stderr}")


def create_modelfile(gguf_path: str, model_name: str = "llama3-qa"):
    """Create Ollama Modelfile for local inference."""
    modelfile_content = f"""FROM {gguf_path}

SYSTEM \"\"\"You are a helpful AI assistant fine-tuned for Q&A tasks. 
Answer questions accurately and concisely based on your training.\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
"""
    modelfile_path = Path("output/Modelfile")
    modelfile_path.write_text(modelfile_content)
    print(f"[Ollama] Modelfile → {modelfile_path}")
    print(f"""
[Ollama] To run locally:
  ollama create {model_name} -f {modelfile_path}
  ollama run {model_name}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LoRA model to GGUF for Ollama")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter", default="output/lora-llama3-qa/lora-adapter")
    parser.add_argument("--merged-dir", default="output/merged-model")
    parser.add_argument("--gguf-out", default="output/model-q4_k_m.gguf")
    parser.add_argument("--quant", default="q4_k_m", choices=["f16", "q8_0", "q4_k_m", "q4_0"])
    parser.add_argument("--model-name", default="llama3-qa")
    args = parser.parse_args()

    # 1. Merge
    merged = merge_adapter(args.base_model, args.adapter, args.merged_dir)

    # 2. Convert
    convert_to_gguf(merged, args.gguf_out, args.quant)

    # 3. Modelfile
    create_modelfile(args.gguf_out, args.model_name)
