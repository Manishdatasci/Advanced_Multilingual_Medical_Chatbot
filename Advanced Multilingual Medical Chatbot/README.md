# 🧠 Advanced Multilingual Medical Chatbot

An AI-powered medical chatbot that supports **English**, **Hindi**, **Marathi**, and **Bengali**.  
It uses **semantic search**, **automatic language detection**, and **confidence scoring** to provide accurate, language-specific medical answers.

> _Developed by Manish Kumar Rajak_

---

## 🚀 Features

- 🌐 Supports **4 languages**: English, Hindi, Marathi, Bengali
- 🔍 Semantic similarity using **sentence embeddings**
- 🧠 Language detection and intelligent switching
- ✅ Displays **confidence score (0–1)** for each response
- 📄 Shows **source of the answer**
- 🔁 Suggests **related questions**
- 🖼️ Clean and user-friendly **Streamlit interface**

---

``` ## 📂 Project Structure

Advanced_Multilingual_Medical_Chatbot/
│
├── main.py # Streamlit app
├── train_model.ipynb # Jupyter Notebook for model training
├── langchain_helper.py # Language handling utilities
├── fixed_qa_dataset.json # Preprocessed Q&A dataset
├── trained_classifier.pkl # Trained classification model
├── embedded_qa_dataset.pkl # Sentence embeddings
├── requirements.txt # Dependencies
└── README.md # Project overview
