import re
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 1) Paths
BASE_MODEL_DIR      = "data/Llama-3.2-3B-Instruct"
LORA_ADAPTER_DIR    = "models/llama-3b-lora"

# 2) Quantization config (4-bit)
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# 3) Load tokenizer + base model
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR, use_fast=False, local_files_only=True
)
model_base = LlamaForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)

# 4) Attach LoRA adapters
model = PeftModel.from_pretrained(
    model_base, LORA_ADAPTER_DIR, device_map="auto", local_files_only=True
)

# 5) Pad token fallback
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 6) List of 12 cipher names
o_algo = ["AES", "DES", "3DES", "Blowfish", "Twofish",
          "Serpent", "Camellia", "CAST-128", "IDEA",
          "RC5", "RC6", "SEED"]

# 7) Build a fully formatted instruction wrapper
instruction = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant that picks the correct block cipher.\n"
    "<</SYS>>\n"
    """You have these block cipher algorithms:
{list}

Requirements:
- Offline
- Feistel-like structure
- Used in academic contexts

Provide exactly one cipher name from the list above. No extra text, no punctuation."""
    " [/INST]"
).format(list=", ".join(o_algo))

# 8) Tokenize + generate with greedy decoding
inputs = tokenizer(instruction, return_tensors="pt", padding=True).to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
)

# 9) Extract the single word answer via regex
raw = tokenizer.decode(out[0], skip_special_tokens=True)
pattern = r"\b(" + "|".join(re.escape(a) for a in o_algo) + r")\b"
match = re.search(pattern, raw)
answer = match.group(1) if match else raw.strip()

print("→", answer)
