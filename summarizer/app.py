# app.py
# Streamlit PDF Summarization Tool using Hugging Face Transformers
# Run: streamlit run app.py

import io
import math
from typing import List, Tuple

import streamlit as st

# Lazy imports (speed up initial load of UI)
def _lazy_imports():
    global PdfReader, pipeline
    from PyPDF2 import PdfReader
    from transformers import pipeline

# ----------------------------
# Text utilities
# ----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text from a PDF uploaded via Streamlit.
    """
    _lazy_imports()
    reader = PdfReader(uploaded_file)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            # Some PDFs can fail; keep app resilient
            continue
    return "\n".join(parts).strip()


def chunk_text(text: str, max_words: int = 500) -> List[str]:
    """
    Simple word-based chunking to fit model context limits.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


@st.cache_resource(show_spinner=False)
def get_summarizer(model_name: str = "facebook/bart-large-cnn"):
    """
    Instantiate a summarization pipeline and cache it across runs.
    """
    _lazy_imports()
    # Note: device_map="auto" requires accelerate; we keep it CPU/GPU-agnostic here.
    return pipeline("summarization", model=model_name)


def summarize_chunks(chunks: List[str], max_length: int = 160, min_length: int = 60, **kwargs) -> List[str]:
    """
    Summarize each chunk individually.
    """
    summarizer = get_summarizer(kwargs.get("model_name", "facebook/bart-large-cnn"))
    summaries = []
    for idx, ch in enumerate(chunks, start=1):
        with st.spinner(f"Summarizing chunk {idx}/{len(chunks)} …"):
            # Guardrail for very small chunks
            ml = max(32, min(max_length, max(64, len(ch.split()) // 2)))
            mn = max(16, min(min_length, ml - 16))
            out = summarizer(ch, max_length=ml, min_length=mn, do_sample=False)
            summaries.append(out[0]["summary_text"])
    return summaries


def summarize_pdf_text(text: str, per_chunk_words: int = 500, final_pass: bool = True) -> Tuple[str, List[str]]:
    """
    End-to-end: chunk the big text, summarize chunks, then (optionally) summarize
    the concatenation of chunk summaries for a final, concise report.
    """
    if not text or not text.strip():
        return "", []

    chunks = chunk_text(text, max_words=per_chunk_words)
    if len(chunks) == 0:
        return "", []

    chunk_summaries = summarize_chunks(chunks)
    joined = " ".join(chunk_summaries)

    if final_pass and len(chunk_summaries) > 1:
        summarizer = get_summarizer()
        with st.spinner("Creating final concise summary …"):
            final = summarizer(joined, max_length=220, min_length=90, do_sample=False)[0]["summary_text"]
    else:
        final = chunk_summaries[0]

    return final, chunk_summaries


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="LLM PDF Summarizer", page_icon="🧠", layout="wide")

st.title("🧠 LLM PDF Summarizer")
st.caption("Upload a PDF and get an AI-generated summary (Hugging Face Transformers).")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Model",
        options=[
            "facebook/bart-large-cnn",
            "sshleifer/distilbart-cnn-12-6",
            "philschmid/bart-large-cnn-samsum",  # another BART variant
            "google/pegasus-xsum",                # different family; may require sentencepiece
            "t5-small",                           # lighter, lower quality
        ],
        index=0,
        help="Choose a summarization model. Default is BART large CNN."
    )
    per_chunk_words = st.slider("Per-chunk words", min_value=200, max_value=1200, value=500, step=50)
    final_pass = st.checkbox("Do a final pass summary", value=True)
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- Large PDFs are summarized in chunks.\n- For faster runs, pick a smaller model.")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf is not None:
    try:
        with st.spinner("Extracting text from PDF …"):
            raw_text = extract_text_from_pdf(uploaded_pdf)

        if not raw_text:
            st.error("Couldn't extract text from this PDF. It might be scanned or image-based. Try OCR first.")
        else:
            st.success("Text extracted.")
            st.write(f"Document length: ~{len(raw_text.split())} words")

            if st.button("Summarize"):
                # Switch model for this run
                get_summarizer.clear()  # clear cache if user changes model
                _ = get_summarizer(model_name)

                final, chunk_summaries = summarize_pdf_text(
                    raw_text,
                    per_chunk_words=per_chunk_words,
                    final_pass=final_pass,
                )

                st.subheader("📌 Final Summary")
                st.write(final)

                with st.expander("See chunk summaries"):
                    for i, s in enumerate(chunk_summaries, start=1):
                        st.markdown(f"**Chunk {i}**")
                        st.write(s)

                st.download_button(
                    label="Download final summary as .txt",
                    data=final,
                    file_name="summary.txt",
                    mime="text/plain",
                )

                # Optionally show extracted text
                with st.expander("Show extracted raw text"):
                    st.text_area("Raw text", raw_text, height=300)
    except Exception as e:
        st.exception(e)
else:
    st.info("Upload a PDF to get started.")
