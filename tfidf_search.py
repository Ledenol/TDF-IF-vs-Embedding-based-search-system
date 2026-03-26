import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocessing

def loaddata(file_path="processed_data.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def getprocessedtexts(data):
    return [item["processed"] for item in data]

def buildtfidf(docs):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix

def search(query, vectorizer, tfidf_matrix, data, top_k=3):
    processed_query = preprocessing(query)
    query_vec = vectorizer.transform([processed_query])

    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        results.append({
            "id": data[idx]["id"],
            "score": float(scores[idx]),
            "original": data[idx]["original"]
        })

    return results