# 📰 Fake News Detector with LIME

This project is a machine learning pipeline for detecting **fake news articles** using **TF-IDF vectorization** and **Logistic Regression**, with **LIME** (Local Interpretable Model-Agnostic Explanations) to explain predictions.

## ✨ Features
- **Binary classification**: Predicts whether a news article is `REAL` or `FAKE`.
- **Interpretable ML**: Highlights words/phrases that influenced the model’s decision using LIME.
- **Streamlit web app**: User-friendly interface to paste or upload articles and see predictions + explanations.
- **Custom preprocessing**: Cleans text by removing URLs, emails, and numbers while preserving semantics.
- **Trained on Kaggle’s Fake/Real News dataset**.
