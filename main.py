import json
from preprocess import preprocessdoc
from tfidf_search import loaddata, getprocessedtexts, buildtfidf, search as tfidf_search
from embedding import getembeddings, search as embeddingsearch

# DATA PATH 
DATA_PATH = r"C:\Users\ASUS\Desktop\NLP\Hybrid_search_engine\Data\documents.txt"

# STEP 1: PREPROCESS
print(" Preprocessing documents...")
preprocessdoc(DATA_PATH)

# STEP 2: LOAD DATA
data = loaddata("processed_data.json")
docs = getprocessedtexts(data)

#  STEP 3: BUILD MODELS
print(" Preparing TF-IDF model...")
vectorizer, tfidf_matrix = buildtfidf(docs)

print(" Preparing embeddings...")
doc_embeddings = getembeddings(docs)

# HYBRID SEARCH FUNCTION
def hybrid_search(query, top_k=3, alpha=0.5):
    """
    alpha = weight for TF-IDF
    (1 - alpha) = weight for embeddings
    """

    tfidf_results = tfidf_search(query, vectorizer, tfidf_matrix, data, top_k=top_k)
    embedding_results = embeddingsearch(query, doc_embeddings, data, top_k=top_k)

    combined_scores = {}

    # TF-IDF contribution
    for res in tfidf_results:
        combined_scores[res["id"]] = alpha * res["score"]

    # Embedding contribution
    for res in embedding_results:
        if res["id"] in combined_scores:
            combined_scores[res["id"]] += (1 - alpha) * res["score"]
        else:
            combined_scores[res["id"]] = (1 - alpha) * res["score"]

    # Sort results
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        results.append({
            "score": score,
            "original": data[doc_id]["original"]
        })

    return results

#  MAIN LOOP 
if __name__ == "__main__":
    print("\nHybrid Search Engine Ready!")

    while True:
        query = input("\nEnter your query (or type 'exit'): ")

        if query.lower() == "exit":
            print(" Exiting...")
            break

        results = hybrid_search(query)

        print("\nTop Results:")
        for r in results:
            print(f"Score: {r['score']:.4f} | {r['original']}")