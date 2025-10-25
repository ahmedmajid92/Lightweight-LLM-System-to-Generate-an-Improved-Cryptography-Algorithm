# pick_with_embeddings.py

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1) Load your full JSON list
algos = json.load(open("data/algorithms.json"))

# 2) For each algorithm, concatenate *all* its field values into one text blob
docs = []
for a in algos:
    # join every value—this gives the model full context
    text = " ".join(str(v) for v in a.values())
    docs.append(text)

# 3) Embed all algorithm‐descriptions
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embs = embedder.encode(docs, convert_to_numpy=True).astype("float32")

# 4) (Re)build a FAISS index in-memory
#    Using inner-product on normalized vectors gives cosine similarity
faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

# 5) Embed your user prompt
prompt = "Suggest an algorithm that is Offline and have a Feistel-like structure and use for telemetry streams."
qv = embedder.encode(prompt, convert_to_numpy=True).astype("float32")
qv = qv.reshape(1, -1)                       # <-- must be 2D
faiss.normalize_L2(qv)

# 6) Search top-1
D, I = index.search(qv.reshape(1, -1), k=1)
best_idx = int(I[0][0])

# 7) Print exactly one algorithm name
print(algos[best_idx]["Algorithm"])
