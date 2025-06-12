import spacy
from nltk.corpus import stopwords

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Test NLP
text = "Python, Java, and Machine Learning are key skills."
doc = nlp(text)

print("Keywords:", [token.text for token in doc if not token.is_stop and token.is_alpha])

# Test NLTK
print("Stopwords:", stopwords.words('english')[:5])  # First 5 stopwords