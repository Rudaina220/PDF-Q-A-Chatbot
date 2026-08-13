import os
import pickle
import tempfile
from pathlib import Path

import faiss
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from pdf_processor import extract_pdf_chunks

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class PDFRAG:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.filename = None
        self.pages = 0
        self.ready = False

    def process_pdf(self, pdf_bytes, filename):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            path = f.name
        try:
            chunks, pages = extract_pdf_chunks(path)
        finally:
            Path(path).unlink(missing_ok=True)

        if not chunks:
            raise ValueError("No readable text was found. Scanned PDFs need OCR.")

        embeddings = self.embedder.encode(
            [x["text"] for x in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = chunks
        self.filename = filename
        self.pages = pages
        self.ready = True
        return {"pages": pages, "chunks": len(chunks)}

    def retrieve(self, question, top_k=5):
        if not self.ready:
            raise ValueError("Please process a PDF first.")

        q = self.embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(q, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                item = self.chunks[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results

    def ask(self, question, top_k=5):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing in your .env file.")

        results = self.retrieve(question, top_k)
        context = "\n\n".join(
            f"[Page {x['page']}]\n{x['text']}" for x in results
        )

        system = (
            "You are a PDF question-answering assistant. "
            "Answer ONLY from the supplied PDF context. "
            "If the answer is not in the context, say it was not found in the PDF. "
            "Be concise and mention page numbers when useful."
        )
        user = f"PDF context:\n\n{context}\n\nQuestion:\n{question}"

        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": 0.2
            },
            timeout=120
        )

        if response.status_code != 200:
            try:
                detail = response.json()["error"]["message"]
            except Exception:
                detail = response.text
            raise RuntimeError(f"Groq API error ({response.status_code}): {detail}")

        return response.json()["choices"][0]["message"]["content"], results
