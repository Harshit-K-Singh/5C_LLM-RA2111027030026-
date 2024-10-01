import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming and Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    processed_tokens = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]
    
    return processed_tokens

def generate_embeddings(processed_texts):
    # Train Word2Vec model
    model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, workers=4)
    
    # Generate embeddings for each word
    embeddings = {word: model.wv[word] for word in model.wv.key_to_index}
    
    return embeddings

def find_top_word_pairs(embeddings, n=100):
    words = list(embeddings.keys())
    vectors = np.array(list(embeddings.values()))
    
    similarity_matrix = cosine_similarity(vectors)
    np.fill_diagonal(similarity_matrix, 0)
    
    # Get top pairs
    indices = np.argsort(similarity_matrix.flatten())[-n:]
    top_pairs = [(words[i // len(words)], words[i % len(words)], similarity_matrix[i // len(words), i % len(words)])
                 for i in indices]
    
    return top_pairs