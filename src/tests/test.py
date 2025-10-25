from llama_cpp import Llama

# 1) Pass n_threads into the constructor
llm = Llama(
    model_path="C:/Users/Ahmed/Desktop/My-Project/models/llama-3-8b-instruct.gguf",
    n_threads=8,         # ← set your desired # of CPU threads here
    verbose=False,        # <-- suppress general logging
    verbose_read=False,   # <-- suppress the metadata dump
)

# 2) Call without n_threads
resp = llm(
    "Suggest a cipher for IoT, 128-bit key, AEAD:", 
    max_tokens=64
)

print(resp["choices"][0]["text"])
