import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import json

# DOWNLOAD NLTK RESOURCES
def download_nltk_resource(resource, path):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

download_nltk_resource('punkt', 'tokenizers/punkt')
download_nltk_resource('stopwords', 'corpora/stopwords')
download_nltk_resource('wordnet', 'corpora/wordnet')
download_nltk_resource('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')

#  INIT 
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# POS TAG CONVERTER
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

#TEXT PREPROCESSING
def preprocessing(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in stop_words and word.isalpha()
    ]

    return ' '.join(tokens)

# LOAD DOCUMENTS FROM FILE
def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]
    return documents

# MAIN PREPROCESS FUNCTION
def preprocessdoc(file_path, output_file="processed_data.json"):
    documents = load_documents(file_path)

    processed_data = []

    for i, text in enumerate(documents):
        processed_text = preprocessing(text)

        processed_data.append({
            "id": i,
            "original": text,
            "processed": processed_text
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)

    print(f" Data saved to {output_file}")

    return processed_data


# OPTIONAL RUN
if __name__ == "__main__":
    FILE_PATH = r"C:\Users\ASUS\Desktop\NLP\Hybrid_search_engine\Data\documents.txt"
    preprocessdoc(FILE_PATH)
