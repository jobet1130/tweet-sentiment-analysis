import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.predictor import predict_sentiment

MODEL_PATH = "models/logreg_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


st.set_page_config(page_title="Tweet Sentiment Classifier", layout="centered")
st.title("Tweet Sentiment Prediction")

tweet = st.text_area("Enter a tweet", height=150)

if st.button("Predict Sentiment"):
    if tweet.strip():
        prediction = predict_sentiment(
            tweet=tweet,
            model_path=MODEL_PATH,
            vectorizer_path=VECTORIZER_PATH,
            encoder_path=ENCODER_PATH
        )
        st.success(f"Predicted Sentiment: **{prediction.capitalize()}**")
    else:
        st.warning("Please enter a tweet.")
