# AI Chatbot for PDFs

Local RAG PDF chatbot using Python, Sentence Transformers, FAISS, Streamlit, and the Groq API.

## Setup

Python 3.10 or 3.11 is recommended.

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Get the key from:
https://console.groq.com/keys

Run:

```cmd
python -m streamlit run app.py
```

## Pipeline

PDF -> text extraction -> chunks -> local embeddings -> FAISS retrieval -> Groq -> answer with sources.

This project uses Groq, not xAI and not Ollama.
