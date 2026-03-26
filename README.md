# TDF-IF-vs-Embedding-based-search-system
🔍 Hybrid Search Engine (TF-IDF + Semantic Embeddings)

A lightweight hybrid search system that combines traditional keyword-based retrieval (TF-IDF) with modern semantic search using transformer embeddings to deliver more accurate and context-aware results.

🚀 Overview

This project implements a dual-search pipeline:

TF-IDF (Lexical Search): Handles exact keyword matching efficiently
Sentence Embeddings (Semantic Search): Captures contextual meaning using transformer models

By combining both approaches, the system can:

Retrieve exact matches
Understand paraphrased queries
Improve relevance across diverse query types
🧠 Key Features
🔎 Hybrid retrieval (TF-IDF + embeddings)
🤖 Semantic understanding using transformer models
⚡ Lightweight and fast implementation
🧩 Modular design (preprocessing, TF-IDF, embeddings separated)
💻 CLI-based query interface
🏗️ Project Structure
Hybrid_search_engine/
│── src/
│   ├── main.py              # Entry point (query interface)
│   ├── preprocess.py        # Data preprocessing
│   ├── tfidf_search.py      # TF-IDF based retrieval
│   ├── embedding.py         # Semantic embedding search
│── documents.txt            # Input dataset
│── processed_data.json      # Preprocessed data
⚙️ How It Works
Preprocessing
Cleans and structures raw text data
Stores processed output for reuse
TF-IDF Search
Converts text into sparse vectors
Computes similarity using cosine similarity
Embedding Search
Uses transformer model (all-MiniLM-L6-v2)
Converts text into dense semantic vectors
Query Handling
User inputs a query
System retrieves results using both methods
Outputs the most relevant matches
🧪 Example Queries

Try queries like:

machine learning
What is AI used for?
neural networks in deep learning
transport systems
fast food
📦 Installation
git clone <repo-link>
cd Hybrid_search_engine
pip install -r requirements.txt
▶️ Run the Project
python Hybrid_search_engine/src/main.py

Then enter:

Enter your query: machine learning
🛠️ Tech Stack
Python
Scikit-learn (TF-IDF)
Sentence Transformers (Hugging Face)
NumPy / Pandas
