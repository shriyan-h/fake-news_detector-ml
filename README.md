# 🔍 Fake News Detector

An NLP-powered web app that detects whether a news article 
is real or fake using Machine Learning.

## 🔗 Live Demo
[Click here to try the app](https://fake-newsdetector-ml-93jwp4zzbwyrdefzvft5qq.streamlit.app/)

## ⚠️ Important Note
This model is trained on American news datasets (Reuters and 
PolitiFact). It works best on US-style news articles. Indian 
or other regional news headlines may be misclassified due to 
differences in writing style.

A news article is classified as **FAKE if confidence > 50%** 
for the fake class.

## 🧠 How It Works
1. Paste a news article or headline
2. Model analyzes the text using NLP
3. Get prediction — REAL or FAKE — with confidence score

## ⚙️ Tech Stack
- Python
- Scikit-learn (Logistic Regression + TF-IDF)
- Streamlit
- Joblib

## 📊 Model Performance
- **Accuracy: 98.67%**
- **Precision: 0.99**
- **Recall: 0.98**
- **F1 Score: 0.99**
- **Dataset: WELFake (72,134 news articles)**

