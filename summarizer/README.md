# LLM PDF Summarizer (Streamlit + Hugging Face)

A simple **PDF summarization** app for SEQATO engineers. Upload a PDF, and get an AI-generated summary using Hugging Face Transformers.

## Features
- 🚀 Streamlit UI for easy use
- 📄 PDF text extraction (PyPDF2)
- ✂️ Automatic chunking for long documents
- 🧠 Summarization via `facebook/bart-large-cnn` (switchable models)
- 💾 Download final summary

## Quickstart

1) **Create and activate a virtual env (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) **Install dependencies**

```bash
pip install -r requirements.txt
```

> The first run will **download the model** weights (~1GB for BART). Use a smaller model like `sshleifer/distilbart-cnn-12-6` if bandwidth is limited.

3) **Run the app**

```bash
streamlit run app.py
```

4) **Open the UI**
- Streamlit will print a local URL (e.g., `http://localhost:8501`). Open it in your browser.
- Upload a PDF and click **Summarize**.

## Tips
- If a PDF is **scanned** (image-only), PyPDF2 can't read text. Run OCR first (e.g., Tesseract, or convert to searchable PDF).
- For faster runs:
  - Use a smaller model (e.g., `sshleifer/distilbart-cnn-12-6`, `t5-small`).
  - Decrease "Per-chunk words" in the sidebar.

## Models
Defaults to `facebook/bart-large-cnn`. You can switch to:
- `sshleifer/distilbart-cnn-12-6` (faster, lighter)
- `google/pegasus-xsum` (may need `sentencepiece`)
- `t5-small` (very light; lower quality)

## Optional: CLI usage
You can also summarize a PDF from the terminal:

```bash
python summarize_pdf.py input.pdf --model facebook/bart-large-cnn --chunk 500 --no-final-pass
```

## Troubleshooting
- **Torch not found / CUDA warnings**: CPU is fine; the app runs without GPU.
- **Model download slow**: Switch to a smaller model or pre-download in CI.
- **UnicodeDecodeError**: Some PDFs have odd encodings; try another extractor like `pdfplumber`.

---

Built for SEQATO's LLM upskilling. Happy summarizing! 🧠
