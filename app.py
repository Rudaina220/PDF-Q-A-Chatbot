import streamlit as st
from dotenv import load_dotenv
from rag import PDFRAG

load_dotenv()

st.set_page_config(page_title="AI PDF Chatbot", page_icon="PDF", layout="wide")
st.title("AI Chatbot for PDFs")
st.caption("Upload a PDF and ask questions using RAG + Groq.")

if "rag" not in st.session_state:
    st.session_state.rag = PDFRAG()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("PDF")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Process PDF", use_container_width=True):
        with st.spinner("Reading and indexing PDF..."):
            try:
                result = st.session_state.rag.process_pdf(uploaded.getvalue(), uploaded.name)
                st.session_state.messages = []
                st.success(f"Processed {result['pages']} pages and {result['chunks']} chunks.")
            except Exception as e:
                st.error(str(e))

    if st.session_state.rag.ready:
        st.success(f"Current PDF: {st.session_state.rag.filename}")
        st.caption(f"{st.session_state.rag.pages} pages | {st.session_state.rag.chunks} chunks")

    top_k = st.slider("Retrieved chunks", 2, 8, 5)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.rag.ready:
    st.info("Upload and process a PDF to start.")
else:
    question = st.chat_input("Ask a question about the PDF...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and asking Groq..."):
                try:
                    answer, sources = st.session_state.rag.ask(question, top_k)
                    st.markdown(answer)
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"**Page {s['page']}** | similarity: {s['score']:.3f}")
                            st.write(s["text"][:800] + "...")
                except Exception as e:
                    answer = f"Error: {e}"
                    st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
