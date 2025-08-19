import streamlit as st
import json
import pickle
import numpy as np
from langdetect import detect, DetectorFactory
from googletrans import Translator
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

DetectorFactory.seed = 0

# --- Load Model and Data ---
with open("trained_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("embedded_qa_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

qa_data = dataset["data"]
embeddings = np.array(dataset["embeddings"])

# --- Load Embedder with Caching ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
translator = Translator()

# --- Supported Languages ---
supported_langs = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali"
}

# --- Language Detection ---
def detect_language(text):
    try:
        if len(text.strip().split()) <= 2:
            return fallback_detect(text)
        lang = detect(text)
        return lang if lang in supported_langs else fallback_detect(text)
    except:
        return fallback_detect(text)

def fallback_detect(text):
    # Bengali script Unicode range
    if any('\u0980' <= ch <= '\u09FF' for ch in text):
        return "bn"
    # Devanagari script Unicode range – Hindi/Marathi fallback
    if any('\u0900' <= ch <= '\u097F' for ch in text):
        return "hi"
    return "en"

# --- Translation Helpers ---
def translate_to_english(text, lang_code):
    if lang_code == "en":
        return text
    try:
        return translator.translate(text, src=lang_code, dest="en").text
    except:
        return text

def translate_to_user_language(text, lang_code):
    if lang_code == "en":
        return text
    try:
        return translator.translate(text, src="en", dest=lang_code).text
    except:
        return text

# --- Core Answer Retrieval ---
def get_top_answer(user_question):
    question_embedding = embedder.encode([user_question])[0]
    similarities = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
    )

    top_idx = np.argmax(similarities)
    top_qa = qa_data[top_idx]

    X = [question_embedding]
    probs = clf.predict_proba(X)[0]
    confidence_score = round(float(np.max(probs)), 2)

    return {
        "question": top_qa["question"],
        "answer": top_qa["answer"],
        "confidence": confidence_score,
        "source": top_qa.get("source", "Unknown"),
        "raw": top_qa
    }

# --- Streamlit App ---
st.set_page_config(page_title="Multilingual Medical Chatbot", layout="wide")
st.title("🩺 Multilingual Medical Chatbot")
st.markdown("Ask your medical question in **English, Hindi, Marathi, or Bengali.**")

# --- Sidebar ---
with st.sidebar:
    st.header("💡 Chatbot Info")
    st.markdown("""
    - Supports **English, Hindi, Marathi, Bengali**
    - Built by **Manish Kumar Rajak** 💻
    - Powered by SentenceTransformer + Logistic Regression
    - Uses Google Translate API for language support
    """)
    show_raw = st.checkbox("🔍 Show raw data info")
    st.markdown("📁 **Dataset:** MedQuAD (by U.S. National Library of Medicine)")

# --- User Input ---
user_input = st.text_input("Type your medical question here 👇")

if user_input:
    user_lang = detect_language(user_input)
    lang_name = supported_langs.get(user_lang, "Unknown")
    translated_question = translate_to_english(user_input, user_lang)
    result = get_top_answer(translated_question)
    final_answer = translate_to_user_language(result["answer"], user_lang)

    st.success(f"🎯 **Answer:** {final_answer}")
    st.info(f"📊 **Confidence:** {result['confidence']} | 🌐 Language: {lang_name}")
    st.caption(f"🔗 Source: {result['source']}")

    if show_raw:
        st.subheader("🧬 Raw Result")
        st.json(result["raw"])

# --- Footer ---
st.markdown("---")
st.markdown("✅ Developed with ❤️ by **Manish Kumar Rajak** | © 2025")
