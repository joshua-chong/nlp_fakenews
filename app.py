
import streamlit as st
from joblib import load
import numpy as np
from pathlib import Path
import streamlit.components.v1 as components

# Import your cleaning + LIME explainer from src
from src.data import simple_clean
from src.explain_lime import lime_explain_text

# ---- Load trained model ----
MODEL_PATH = Path("models/pipe_lr.joblib")
if not MODEL_PATH.exists():
    st.error("❌ Model not found. Please train and save it first as models/fake_news_lr.joblib")
    st.stop()

pipe = load(MODEL_PATH)
vec = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]

# ---- Streamlit App Layout ----
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector")
st.write("Paste an article and press **Analyze**. "
         "The model (TF-IDF + Logistic Regression) will predict REAL/FAKE "
         "and highlight influential words using LIME.")

# Sidebar
st.sidebar.title("Settings")
apply_cleaning = st.sidebar.checkbox("Apply simple_clean preprocessing", value=True)
show_lime = st.sidebar.checkbox("Generate LIME explanation", value=True)
top_k = st.sidebar.slider("Top contributing terms", 5, 30, 10)
num_features = st.sidebar.slider("LIME features", 5, 20, 10)

# Input area
default_text = "Breaking: Scientists guarantee a miracle cure in 24 hours! Click to claim your free kit now."
text = st.text_area("Enter article text:", value=default_text, height=250)

if st.button("🔎 Analyze") and text.strip():
    # Preprocess input
    input_text = simple_clean(text) if apply_cleaning else text

    # Prediction
    prob_real = float(pipe.predict_proba([input_text])[0, 1])
    pred = "REAL ✅" if prob_real >= 0.5 else "FAKE ❌"
    st.subheader(f"Prediction: {pred}")
    st.write(f"**p(real):** {prob_real:.3f} | **p(fake):** {1 - prob_real:.3f}")

    # Top contributing terms
    X = vec.transform([input_text])
    coefs = clf.coef_[0]
    indices = X.nonzero()[1]
    contrib = [(vec.get_feature_names_out()[j], float(X[0, j] * coefs[j])) for j in indices]
    contrib = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:top_k]

    st.markdown("#### Top contributing terms")
    st.dataframe({"term": [t for t, _ in contrib], "weight": [round(w, 4) for _, w in contrib]})

    # LIME explanation
    if show_lime:
        st.markdown("#### LIME explanation")
        exp = lime_explain_text(pipe, input_text, num_features=num_features)
        components.html(exp.as_html(), height=600, scrolling=True)
