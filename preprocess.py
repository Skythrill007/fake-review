import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-fitted TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer')

# Define preprocessing functions
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and perform stemming
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Join tokens back into text
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

def preprocess_input(text):
    # Apply the same preprocessing steps as used for training data
    preprocessed_text = preprocess_text(text)
    # Convert preprocessed text to TF-IDF vector using the loaded vectorizer
    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
    return tfidf_vector
