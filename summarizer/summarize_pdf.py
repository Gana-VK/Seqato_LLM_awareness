# summarize_pdf.py
# CLI tool to summarize a PDF. Usage:
# python summarize_pdf.py input.pdf --model facebook/bart-large-cnn --chunk 500

import argparse
from typing import List

from PyPDF2 import PdfReader
from transformers import pipeline

def extract_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts).strip()

def chunk_text(text: str, max_words: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words) if words[i:i+max_words]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Input PDF path")
    ap.add_argument("--model", default="facebook/bart-large-cnn", help="HF model id")
    ap.add_argument("--chunk", type=int, default=500, help="Words per chunk")
    ap.add_argument("--final-pass", dest="final_pass", action="store_true")
    ap.add_argument("--no-final-pass", dest="final_pass", action="store_false")
    ap.set_defaults(final_pass=True)
    args = ap.parse_args()

    text = extract_text(args.pdf)
    if not text:
        print("No text extracted. Is this a scanned (image) PDF?")
        return

    chunks = chunk_text(text, max_words=args.chunk)
    summarizer = pipeline("summarization", model=args.model)

    summaries = []
    for i, ch in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] Summarizing chunk…")
        out = summarizer(ch, max_length=160, min_length=60, do_sample=False)
        summaries.append(out[0]["summary_text"])

    final = " ".join(summaries)
    if args.final_pass and len(summaries) > 1:
        final = summarizer(final, max_length=220, min_length=90, do_sample=False)[0]["summary_text"]

    print("\n=== FINAL SUMMARY ===\n")
    print(final)

if __name__ == "__main__":
    main()
