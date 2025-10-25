# scripts/infer_server.py
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 1) Paths
BASE_MODEL_DIR      = "data/Llama-3.2-3B-Instruct"
LORA_ADAPTER_DIR    = "models/llama-3b-lora"

# 2) Quantization config
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# 3) Load once at import
print("Loading tokenizer…", end="", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR, use_fast=False, local_files_only=True
)
print(" done.")
print("Loading base model…", end="", flush=True)
model_base = LlamaForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)
print("Attaching LoRA…", end="", flush=True)
model = PeftModel.from_pretrained(
    model_base, LORA_ADAPTER_DIR, device_map="auto", local_files_only=True
)
print(" done. Ready to infer!")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def infer(prompt: str) -> str:
    algos = ["AES", "DES", "3DES", "Blowfish", "Twofish",
             "Serpent", "Camellia", "CAST-128", "IDEA",
             "RC5", "RC6", "SEED"]
    algo_list_str = ", ".join(algos)
    full_prompt = f"""\
You have these block cipher algorithms:
{algo_list_str}

Suggest an algorithm that is Offline and have a Feistel-like structure and uses for academic interest.
"""
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
    out    = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return text.splitlines()[-1]

if __name__ == "__main__":
    # simple REPL
    print("Enter your own prompts (empty line to quit)")
    while True:
        q = input("> ").strip()
        if not q:
            break
        print("→", infer(q))
