import pandas as pd
import nltk
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

def train_model():
    data_path = 'dataset.csv'
    df = pd.read_csv(data_path)
    df['processed_question'] = df['question'].apply(preprocess_text)
    
    # Improved TF-IDF: word ngrams + char ngrams for better keyword capture
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            analyzer='word',
            sublinear_tf=True
        )),
        ('classifier', MultinomialNB(alpha=0.1))  # Better smoothing
    ])
    
    pipeline.fit(df['processed_question'], df['category'])
    joblib.dump(pipeline, 'question_classifier.joblib')
    print("Model trained and saved as 'question_classifier.joblib'")
    return pipeline

def predict_question(question):
    model_path = 'question_classifier.joblib'
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print("No saved model found. Training new model...")
        model = train_model()
    
    processed = preprocess_text(question)
    prediction = model.predict([processed])[0]
    
    # Log for debugging
    print(f"Input: '{question}' -> Processed: '{processed}' -> Predicted: '{prediction}'")
    
    return prediction

# Auto-train
if not os.path.exists('question_classifier.joblib'):
    train_model()

