import streamlit as st
import json
import pickle
import numpy as np
from langdetect import detect, DetectorFactory
from googletrans import Translator
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

DetectorFactory.seed = 0

# Load model and data
with open("trained_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("embedded_qa_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

qa_data = dataset["data"]
embeddings = np.array(dataset["embeddings"])

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
translator = Translator()

# Supported languages
supported_langs = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali"
}

def detect_language(text):
    try:
        if len(text.strip().split()) <= 2:
            # Fallback to asking user or default if too short
            return fallback_detect(text)
        lang = detect(text)
        return lang if lang in supported_langs else fallback_detect(text)
    except:
        return fallback_detect(text)

def fallback_detect(text):
    # Heuristic-based Bengali detection (basic example)
    bengali_chars = set("অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ")
    if any(char in bengali_chars for char in text):
        return "bn"
    # Could be Hindi/Marathi if Devnagari
    devanagari_range = ('\u0900', '\u097F')
    if any(devanagari_range[0] <= ch <= devanagari_range[1] for ch in text):
        return "hi"  # defaulting to Hindi
    return "en"

def translate_to_english(text, lang_code):
    if lang_code == "en":
        return text
    try:
        return translator.translate(text, src=lang_code, dest="en").text
    except:
        return text  # fallback

def translate_to_user_language(text, lang_code):
    if lang_code == "en":
        return text
    try:
        return translator.translate(text, src="en", dest=lang_code).text
    except:
        return text

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

user_input = st.text_input("Type your medical question here 👇")

if user_input:
    # For Detecting language
    user_lang = detect_language(user_input)
    lang_name = supported_langs.get(user_lang, "Unknown")

    # For Translating question to English
    translated_question = translate_to_english(user_input, user_lang)

    # For Getting answer
    result = get_top_answer(translated_question)

    # For Translating answer back
    final_answer = translate_to_user_language(result["answer"], user_lang)

    # For Displaying result
    st.success(f"🎯 **Answer:** {final_answer}")
    st.info(f"📊 **Confidence:** {result['confidence']} | 🌐 Language: {lang_name}")
    st.caption(f"🔗 Source: {result['source']}")

    if show_raw:
        st.subheader("🧬 Raw Result")
        st.json(result["raw"])
