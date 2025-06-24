# 🧠 Sentiment Analysis on Tweets

This project performs sentiment analysis on tweets using classic machine learning techniques such as TF-IDF vectorization and Logistic Regression or SVM classifiers. It is built as an intermediate-level NLP project, ideal for learners or practitioners exploring text classification and NLP pipelines.

---

## 📌 Project Structure

```
tweet-sentiment-analysis/
├── data/                        # Raw and processed datasets
├── notebooks/                   # Step-by-step Jupyter notebooks
├── src/                         # Reusable Python modules
├── models/                      # Saved models and vectorizers
├── app/                         # Streamlit app (optional UI)
├── reports/                     # Visuals and evaluation results
├── main.py                      # CLI pipeline runner
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 🗂️ Notebooks Overview

| Notebook                             | Purpose                                             |
|--------------------------------------|-----------------------------------------------------|
| `01_data_loading_and_exploration`    | Load dataset and explore sentiment distribution     |
| `02_text_preprocessing`              | Clean and normalize tweet text                      |
| `03_feature_extraction_tfidf`        | Convert text to numeric using TF-IDF                |
| `04_model_training_logreg_svm`       | Train Logistic Regression / SVM models              |
| `05_model_evaluation_metrics`        | Evaluate model with accuracy, confusion matrix, etc |
| `06_single_tweet_prediction`         | Predict sentiment of a new tweet                    |

---

## 🧰 Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Make sure to download NLTK data in your scripts:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

## 📊 Model Overview

- **Text Vectorization:** TF-IDF
- **Models Used:** Logistic Regression, Support Vector Machines (SVM)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix

---

## 🚀 How to Use

### ▶️ Run Jupyter Notebooks
Use each notebook in sequence to train and evaluate the model.

### 📲 Predict Sentiment for New Tweets

In `06_single_tweet_prediction.ipynb` or via the optional app:

```python
from src.predictor import predict_sentiment
predict_sentiment("I love this product!")
```

### 🌐 Run the Streamlit App (Optional)

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Sample Output

- ✅ Accuracy: 85–90% (TF-IDF + LogReg on balanced dataset)
- 📉 Confusion matrix and word clouds included in `reports/figures/`

---

## 📦 Dataset

- [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## 🧠 Future Enhancements

- Extend to **BERT-based sentiment analysis**
- Use real-time data from **Twitter API (X API)**
- Store predictions in a **PostgreSQL** or **SQLite** database
- Deploy via **Docker** or **FastAPI**

---

## 🤝 Contributing

Pull requests and improvements are welcome! Please fork the repo and submit your changes.

---

## 📄 License

This project is licensed under the MIT License.

---

## ✨ Acknowledgments

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Inspired by classic ML pipelines and NLP tutorials
