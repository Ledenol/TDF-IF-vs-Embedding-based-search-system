import json
from tfidf_search import loaddata, getprocessedtexts, buildtfidf, search as tfidf_search
from embedding import getembeddings, search as embedding_search
from main import hybrid_search

DATA_PATH = r"C:\Users\ASUS\Desktop\NLP\Hybrid_search_engine"

# Load processed data
data = loaddata(DATA_PATH + r"\processed_data.json")

# Load evaluation queries
with open(DATA_PATH + r"\Data\evaluation_queries.json", "r") as f:
    queries = json.load(f)

data = loaddata(DATA_PATH + r"\processed_data.json")

docs = getprocessedtexts(data)   # REQUIRED

vectorizer, tfidf_matrix = buildtfidf(docs)

doc_embeddings = getembeddings(docs)

# Load evaluation queries
with open(DATA_PATH + r"\Data\evaluation_queries.json", "r") as f:
    queries = json.load(f)

def precision_at_k(results, relevant_docs, k=3):
    retrieved_ids = [r["id"] for r in results[:k]]
    relevant_retrieved = sum([1 for doc_id in retrieved_ids if doc_id in relevant_docs])
    return relevant_retrieved / k


def evaluate():
    tfidf_scores = []
    embedding_scores = []
    hybrid_scores = []

    for q in queries:
        query = q["query"]
        relevant = q["relevant_docs"]

        tfidf_res = tfidf_search(query, vectorizer, tfidf_matrix, data)
        emb_res = embedding_search(query, doc_embeddings, data)
        hybrid_res = hybrid_search(query)

        tfidf_scores.append(precision_at_k(tfidf_res, relevant))
        embedding_scores.append(precision_at_k(emb_res, relevant))
        hybrid_scores.append(precision_at_k(hybrid_res, relevant))

    print("\n=== Evaluation Results ===")
    print(f"TF-IDF Precision@3: {sum(tfidf_scores)/len(tfidf_scores):.4f}")
    print(f"Embedding Precision@3: {sum(embedding_scores)/len(embedding_scores):.4f}")
    print(f"Hybrid Precision@3: {sum(hybrid_scores)/len(hybrid_scores):.4f}")


if __name__ == "__main__":
    evaluate()