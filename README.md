# Question Classification System 🧠

A ML-powered web app that classifies questions into categories (Person, Location, Time, Definition, Reason, Method) using NLP + Naive Bayes.

## Features
- **NLP Preprocessing**: Lowercase, punctuation removal, tokenization, stopwords removal (NLTK)
- **Feature Extraction**: TF-IDF
- **ML Model**: Multinomial Naive Bayes (scikit-learn)
- **Web Interface**: Flask backend + HTML/CSS frontend
- **Dataset**: 100+ labeled questions in `dataset.csv`

## Quick Start (Windows)

1. **Create & Activate Virtual Environment**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Download NLTK Data** (first time only)
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

4. **Run the App**
   ```
   python app.py
   ```

5. **Open Browser**
   Visit `http://127.0.0.1:5000/`

## Test Samples
| Question | Expected Category |
|----------|-------------------|
| Who is Elon Musk? | Person |
| Where is Taj Mahal? | Location |
| When was Python created? | Time |
| What is machine learning? | Definition |
| Why does it rain? | Reason |
| How to make tea? | Method |

## Project Structure
```
question-classifier/
├── app.py              # Flask app
├── model.py            # ML model + NLP
├── dataset.csv         # Training data
├── requirements.txt    # Dependencies
├── README.md           # This file
├── question_classifier.joblib  # Saved model (auto-created)
├── templates/
│   └── index.html      # Frontend
└── static/
    └── style.css       # Styles
```

## Model Training
- Runs automatically on first startup (`model.py`)
- Retrains if `question_classifier.joblib` missing
- Uses `dataset.csv` with TF-IDF (5000 features, n-grams)

Built with ❤️ using Python, scikit-learn, NLTK, Flask.

