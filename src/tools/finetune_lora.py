from transformers import LlamaForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch, os

if not torch.cuda.is_available():
    raise RuntimeError("CUDA device not found!")

os.environ["TORCHDYNAMO_BACKEND"] = "eager"
os.environ["TORCHINDUCTOR"]     = "0"

MODEL_ID   = "/mnt/c/Users/Ahmed/Desktop/My-Project/data/llama-3.2-3b-instruct"
OUTPUT_DIR = "/mnt/c/Users/Ahmed/Desktop/My-Project/models/llama-3b-lora"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, use_fast=False, local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = LlamaForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=False,
    local_files_only=True,
)
# **IMPORTANT** disable KV‐cache so loss is computed
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# load and tokenize
ds = load_dataset(
    "json",
    data_files="/mnt/c/Users/Ahmed/Desktop/My-Project/data/block_cipher_dataset.jsonl",
)["train"]

def tokenize_fn(ex):
    tok = tokenizer(
        ex["instruction"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    # pass labels = input_ids for Trainer loss
    tok["labels"] = tok["input_ids"].copy()
    return tok

ds = ds.map(tokenize_fn, batched=True)

# trainer args
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=10,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    torch_compile=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
