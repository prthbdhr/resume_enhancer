import re

from typing import List  # Add this at the top
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords_tfidf(text: str, top_n: int = 15) -> List[str]:
    text_clean = re.sub(r'[^\w\s]', '', text.lower())

    # Include bigrams (1-2 word phrases)
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # <-- Now captures phrases
        max_features=100
    )

    X = vectorizer.fit_transform([text_clean])
    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    top_indices = scores.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]