from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once (VERY IMPORTANT)
model = SentenceTransformer('all-MiniLM-L6-v2')

def getembeddings(docs):
    return model.encode(docs, convert_to_numpy=True)

def embedquery(query):
    return model.encode([query], convert_to_numpy=True)[0]

def cosine_similarity_search(query_vec, doc_embeddings, data, top_k=3):
    # normalize vectors
    query_vec = query_vec / np.linalg.norm(query_vec)
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    # compute similarity
    scores = np.dot(doc_embeddings, query_vec)

    # rank
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "id": data[idx]["id"],
            "score": float(scores[idx]),
            "original": data[idx]["original"]
        })

    return results

def search(query, doc_embeddings, data, top_k=3):
    query_vec = embedquery(query)
    return cosine_similarity_search(query_vec, doc_embeddings, data, top_k)