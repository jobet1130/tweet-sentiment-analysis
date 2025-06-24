from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from typing import List, Tuple

def create_vectorizer(
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.9,
    sublinear_tf: bool = True,
    norm: str = 'l2'
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        norm=norm
    )

def fit_vectorizer(texts: List[str], output_path: str) -> TfidfVectorizer:
    vectorizer = create_vectorizer()
    vectorizer.fit(texts)
    joblib.dump(vectorizer, output_path)
    return vectorizer

def transform_texts(texts: List[str], vectorizer_path: str):
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer.transform(texts)
