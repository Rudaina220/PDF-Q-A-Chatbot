from pypdf import PdfReader

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def clean_text(text):
    return " ".join(text.replace("\x00", " ").split())

def split_text(text):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - CHUNK_OVERLAP
    return chunks

def extract_pdf_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    for page_no, page in enumerate(reader.pages, 1):
        text = clean_text(page.extract_text() or "")
        for chunk in split_text(text):
            chunks.append({"page": page_no, "text": chunk})
    return chunks, len(reader.pages)
