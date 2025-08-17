# 🧠 Advanced Multilingual Medical Chatbot

An AI-powered medical chatbot that supports **English**, **Hindi**, **Marathi**, and **Bengali**.  
It uses **semantic search**, **automatic language detection**, **Google Translate**, and **confidence scoring** to deliver accurate medical answers.

> _Developed by Manish Kumar_

---

## 🚀 Features

- 🌐 Supports 4 languages: English, Hindi, Marathi, Bengali
- 🔍 Semantic similarity search using SentenceTransformer
- 🧠 Automatic language detection with fallback logic
- 🔁 Automatic translation (question and answer)
- 📊 Shows confidence score (0–1)
- 📄 Displays answer source
- 🔎 Optional view of raw QA data
- 🖼️ Clean, interactive UI built with Streamlit

---

``` ## 📂 Project Structure

Advanced_Multilingual_Medical_Chatbot/
│
├── main.py # Streamlit app interface
├── train_model.ipynb # Model training and evaluation
├── langchain_helper.py # (optional future: helper functions for LangChain or modular logic)
├── fixed_qa_dataset.json # Preprocessed Q&A data from MedQuAD
├── trained_classifier.pkl # Saved Logistic Regression model
├── embedded_qa_dataset.pkl # Embedded Q&A dataset using SentenceTransformer
├── requirements.txt # Required Python packages
└── README.md # Project documentation


