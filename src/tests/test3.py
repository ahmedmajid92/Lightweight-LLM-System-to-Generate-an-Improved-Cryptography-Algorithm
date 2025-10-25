import json
import re
from llama_cpp import Llama

# 1) Load algorithm names
algos = json.load(open("/mnt/c/Users/Ahmed/Desktop/My-Project/data/algorithms.json"))
names = [a["Algorithm"] for a in algos]

# 2) Numbered list
options = "\n".join(f"{i+1}. {name}" for i, name in enumerate(names))

# 3) Instantiate with larger context
llm = Llama(
    model_path="/mnt/c/Users/Ahmed/Desktop/My-Project/models/llama-3.2-3b-instruct.gguf",
    n_threads=8,
    n_ctx=2048,
    verbose=False,
    verbose_read=False,
)

# 4) Clear, asterisk-free prompt
prompt = f"""Here are 12 block cipher options:
{options}

Question: Suggest an algorithm that is Offline and have a Feistel-like structure and use for telemetry streams.
Please reply with the integer only (for example: 3), no extra characters.
"""

# 5) Call model
resp = llm(prompt, max_tokens=4)
text = resp["choices"][0]["text"].strip()

# 6) Extract first integer found
m = re.search(r"\d+", text)
if not m:
    raise ValueError(f"Could not parse an integer from model response: {text!r}")
num = int(m.group(0))

# 7) Print the corresponding algorithm name
print(names[num-1])
