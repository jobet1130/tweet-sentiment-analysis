from src.preprocessing import preprocess
from sklearn.base import BaseEstimator
import joblib
from typing import Union

def load_model(path: str) -> BaseEstimator:
    return joblib.load(path)

def load_vectorizer(path: str):
    return joblib.load(path)

def load_encoder(path: str):
    return joblib.load(path)

def predict_sentiment(tweet: str,
                      model_path: str,
                      vectorizer_path: str,
                      encoder_path: str) -> Union[str, int]:
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)
    encoder = load_encoder(encoder_path)

    cleaned = preprocess(tweet)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return encoder.inverse_transform(prediction)[0]
