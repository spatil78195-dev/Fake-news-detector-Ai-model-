"""Download NLTK data during Render/Railway build."""
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
print("NLTK data ready.")
