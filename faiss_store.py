import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


#  Load Data 
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Get Original Text
def get_original_texts(data):
    return [item["original"] for item in data]


#  Build FAISS Index 
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index


#Search 
def search(query, model, index, data, top_k=3):
    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "score": float(distances[0][i]),
            "original": data[idx]["original"]
        })

    return results


#  Main
if __name__ == "__main__":
    file_path = "data/processed_data.json"

    data = load_data(file_path)
    documents = get_original_texts(data)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(documents).astype("float32")

    index = build_faiss_index(embeddings)

    query = input("Enter your query: ")

    results = search(query, model, index, data)

    print("\nTop Results:")
    for r in results:
        print(f"Score: {r['score']:.4f} | Text: {r['original']}")