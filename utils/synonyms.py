from typing import List

SYNONYM_MAP = {
    "py": "python",
    "js": "javascript",
    "ml": "machine learning",
    "ai": "artificial intelligence"
}

def normalize_terms(terms: List[str]) -> List[str]:
    return [SYNONYM_MAP.get(term.lower(), term) for term in terms]