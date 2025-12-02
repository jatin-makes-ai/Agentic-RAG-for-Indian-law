#!/usr/bin/env python3
"""
pdf_to_text.py

Extracts text from a PDF and writes a cleaned .txt file.

Usage:
    python scripts/pdf_to_text.py --pdf_path data/raw/Constitution/constitution.pdf --out_dir data/text

Outputs:
    data/text/constitution.txt
"""

import argparse
import os
import re
from pathlib import Path

import pdfplumber

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path: str) -> (str, int):
    """
    Extract text from PDF using pdfplumber.
    Returns the full text and number of pages processed.
    Each page is prefixed with a page marker: [[PAGE n]]
    """
    texts = []
    page_count = 0
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            page_count += 1
            # extract_text may return None for some pages; fallback to ""
            txt = p.extract_text(x_tolerance=2) or ""
            # Some PDFs include weird control chars; normalize line endings
            txt = txt.replace("\r\n", "\n").replace("\r", "\n")
            # add a clear page delimiter (useful for downstream provenance)
            texts.append(f"\n\n[[PAGE {p.page_number}]]\n{txt}\n")
    full = "\n".join(texts)
    # basic normalization: collapse >2 newlines to exactly 2
    full = re.sub(r"\n{3,}", "\n\n", full)
    # strip leading/trailing whitespace
    full = full.strip() + "\n"
    return full, page_count

def write_text_file(text: str, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

def main(pdf_path: str, out_dir: str):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} not found")

    print(f"Extracting text from: {pdf_path}")
    text, pages = extract_text_from_pdf(str(pdf_path))
    print(f"Extracted text from {pages} pages; total characters: {len(text)}")

    out_file = Path(out_dir) / (pdf_path.stem.lower().replace(" ", "_") + ".txt")
    write_text_file(text, str(out_file))
    print(f"Wrote cleaned text to: {out_file}")

    # print a short sample (first 800 chars) so you can sanity-check quickly
    sample = text[:800].replace("\n", "\\n")
    print("\n--- SAMPLE (first 800 chars, \\n = newline) ---")
    print(sample)
    print("\n--- End sample ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDF to single cleaned .txt")
    parser.add_argument("--pdf_path", required=True, help="Path to PDF file")
    parser.add_argument("--out_dir", default="data/text", help="Directory to write output .txt")
    args = parser.parse_args()
    main(args.pdf_path, args.out_dir)
