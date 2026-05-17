import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('fake_news_mod.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

model, vectorizer = load_model()

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict
def predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = int(model.predict(vector)[0])
    probability = model.predict_proba(vector)[0]
    confidence = round(float(max(probability)) * 100, 2)
    label = "REAL" if prediction == 1 else "FAKE"
    return label, confidence

# UI
st.title("🔍 Fake News Detector")
st.markdown("Paste a news article or headline to check if it is real or fake.")
st.divider()

news_input = st.text_area("Paste news article or headline here", height=200)

if st.button("Analyze News", use_container_width=True):
    if news_input.strip():
        label, confidence = predict(news_input)
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Prediction", label)
        c2.metric("Confidence", str(confidence) + "%")
        st.progress(confidence / 100)
        if label == "FAKE":
            st.error("This article appears to be FAKE news.")
        else:
            st.success("This article appears to be REAL news.")
    else:
        st.warning("Please paste some text first.")

st.divider()
st.caption("Model: Logistic Regression | Accuracy: 98.67% | Dataset: WELFake")