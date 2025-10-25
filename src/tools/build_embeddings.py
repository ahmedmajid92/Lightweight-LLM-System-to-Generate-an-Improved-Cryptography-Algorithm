import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_embeddings(json_path: str, index_path: str):
    with open(json_path) as f:
        algos = json.load(f)
    texts = []
    for a in algos:
        summary = (
            f"{a['Algorithm']} is a {a.get('Type','')} block cipher "
            f"with block size {a.get('Block Size','')}, key size {a.get('Key Size','')}, "
            f"structure {a.get('Structure','')}, speed {a.get('Speed','')} and security {a.get('Security','')}."
        )
        texts.append(summary)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode(texts)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs.astype('float32'))

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    # Save metadata
    import pickle
    with open('data/algos_meta.pkl', 'wb') as f:
        pickle.dump({'names': [a['Algorithm'] for a in algos]}, f)

if __name__ == '__main__':
    build_embeddings('data/algorithms.json', 'data/algos.index')