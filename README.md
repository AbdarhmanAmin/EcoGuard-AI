# EcoGuard-AI
<div align="center">

# 🌊 EcoGuard AI
### Water Intelligence Platform

An AI-powered platform for predicting water leakage, detecting contamination, and answering water-quality questions using RAG.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-06b6d4?style=for-the-badge&logo=streamlit&logoColor=white)](https://ecoguard-ai-gbbkp69xzs8onfz7zezkat.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#license)

**[🚀 Try the Live App](https://ecoguard-ai-gbbkp69xzs8onfz7zezkat.streamlit.app)**

</div>

---

## 📌 About

**EcoGuard AI** brings together three AI systems in one clean interface:

- 💧 **Leakage Prediction** — SVM model that detects potential water leaks from sensor data
- 🧪 **Contamination Prediction** — XGBoost model that classifies water as safe/unsafe
- 🤖 **Water Assistant (RAG)** — a chatbot that answers water-quality questions using official reference documents + Llama 3.3 70B

Built for anyone who needs fast, reliable answers about water systems — combining structured ML predictions with document-grounded AI in one place.

---

## ✨ Features

- **Instant predictions** for leakage and contamination with clear, color-coded results
- **AI chat assistant** that answers from your documents, with sources cited for every answer (document + page number)
- **Falls back on general knowledge** when a question isn't covered by the documents — never leaves you without an answer
- **One-click knowledge base rebuild** — re-index PDFs without redeploying
- **Clean, professional UI** with dark theme and smooth navigation between all three modules

---

## 🛠️ Tech Stack

`Streamlit` · `Scikit-learn (SVM)` · `XGBoost` · `LangChain` · `ChromaDB` · `HuggingFace Embeddings` · `Groq (Llama 3.3 70B)`

---

## ⚙️ How It Works

**Leakage / Contamination:** sensor inputs → scaled → passed to the trained model → safe/unsafe verdict.

**Water Assistant (RAG):**
1. PDFs are split into chunks and embedded into a Chroma vector database
2. A user's question retrieves the most relevant chunks
3. Retrieved context + question are sent to Llama 3.3 70B via Groq
4. The model answers using the documents when relevant, or its own knowledge otherwise
5. Sources are shown alongside every answer

---

## 🚀 Getting Started

```bash
git clone https://github.com/abdarhmanamin/ecoguard-ai.git
cd ecoguard-ai
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file with your free [Groq API key](https://console.groq.com):

```env
GROQ_API_KEY="your-groq-api-key-here"
```

> 🔒 Never commit your `.env` file. On Streamlit Cloud, add secrets via **App Settings → Secrets**.

---

## 📂 Project Structure

```
EcoGuard-AI/
├── app.py
├── requirements.txt
├── runtime.txt
├── Model 1 _ Water Leakage/       # SVM model + scaler
├── Model 2 _Water Contamination/  # XGBoost model + scaler
└── Model 3 _ RAG/
    ├── documents/                 # Reference PDFs
    └── vector_db/                 # Chroma vector store
```

---

## 📬 Contact

**Abdarhman Amin** — [GitHub](https://github.com/abdarhmanamin) · [Live App](https://ecoguard-ai-gbbkp69xzs8onfz7zezkat.streamlit.app)

## 📄 License

MIT — free to use, modify, and build upon with attribution.

<div align="center">

**Made with 💧 and AI.**

</div>
