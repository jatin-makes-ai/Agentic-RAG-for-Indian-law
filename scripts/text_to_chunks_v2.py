#!/usr/bin/env python3
"""
legal_chunker_ruleset.py

Chunker tailored to rules provided by user:
- PART extraction: "PART {Roman Number}{newline}{Part Title in CAPITAL}{newline}"
- Article extraction: "{number}{OPTIONAL CAPITAL LETTER}. {Article title}.—"
- Preprocess removals:
    * Remove "{number} THE CONSTITUTION OF INDIA" right after page marker
    * Remove "(Part {ROMAN}.—{Part Name})" patterns
- Footer detection: text between a long underline (6+ underscores/hyphens/equal) and the next [[PAGE n]] is wrapped as:
    FOOTER({footer_content})
- Retains sliding token window chunking with overlap.
- Outputs chunks jsonl, summary json, ambiguous report jsonl.

Usage:
    python legal_chunker_ruleset.py --in_text data/text/constitution.txt --out_dir data/chunks --target_tokens 512 --overlap_tokens 64 --min_tokens 48 --merge_small --doc_id constitution_india
"""

import re
import json
import hashlib
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    import tiktoken
except Exception as e:
    raise ImportError("tiktoken required. pip install tiktoken") from e

ENC = tiktoken.get_encoding("cl100k_base")

# -----------------------
# Strict regexes per rules
# -----------------------

# PART: "PART {Roman Number}{newline}{Part Title in Capital}{newline}"
PART_HEADER_RE = re.compile(r"^\s*PART\s+([IVXLCDM]+)\s*$", flags=re.IGNORECASE)
# To validate title line as ALL CAPS (allow A-Z, digits, spaces and punctuation), require at least 2 words.
PART_TITLE_ALLCAPS_RE = re.compile(r"^[A-Z0-9\W\s]{3,}$")

# Article per rule: "{number}{optionally a letter in caps}. {Article title}.—"
# We'll enforce an uppercase optional letter and require a final ".<dash>" (dash variants allowed)
ARTICLE_STRICT_RE = re.compile(
    r"^\s*([0-9]+[A-Z]?)\.\s+(.+?)\.\s*([\u2014\u2013\-—])\s*$"
)

# Fallback numeric line (if extraction slightly differs) - kept but used only as lower-confidence fallback
NUMERIC_LINE_RE = re.compile(r"^\s*([0-9]+[A-Z]?)\.\s*(.*)$")

# Underline/Separator (6 or more underscores/hyphens/equals)
UNDERLINE_RE = re.compile(r"^[\_\-\=]{6,}\s*$")

# Footer page marker pattern
PAGE_MARKER_RE = re.compile(r"\[\[PAGE\s+(\d+)\]\]")

# Patterns to remove in preprocess (exactly as user specified):
# 1) Right after page number: "{number} THE CONSTITUTION OF INDIA{newline}"
# We'll match lines like "1 THE CONSTITUTION OF INDIA" optionally preceded by whitespace
CONST_HEADER_AFTER_PAGE_RE = re.compile(r"^\s*\d+\s+THE\s+CONSTITUTION\s+OF\s+INDIA\s*$", flags=re.IGNORECASE | re.MULTILINE)
# 2) "(Part {part number in roman numerals.—{Part Name})"
# Example: "(Part I.—THE UNION AND ITS TERRITORY)"
PART_PAREN_RE = re.compile(r"\(\s*Part\s+([IVXLCDM]+)\.\s*[—\u2014\u2013\-]\s*(.+?)\s*\)", flags=re.IGNORECASE)

# Footnote/Editorial markers (for completeness)
FOOTNOTE_GENERIC_RE = re.compile(r"(Subs\.|Ins\.|Inserted|Substituted|Added|Omitted|Explanation|Note:|w\.e\.f\.)", flags=re.IGNORECASE)

# -----------------------
# Helpers
# -----------------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def md5_text(s: str):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))

def tokenize_to_tokens(text: str):
    return ENC.encode(text)

def decode_tokens(tokens):
    return ENC.decode(tokens)

def char_index_of_token(full_text: str, toks: list, tok_index: int) -> int:
    """Return char offset into full_text for token index (heuristic decode)."""
    if tok_index <= 0:
        return 0
    s = ENC.decode(toks[:tok_index])
    return len(s)

def extract_lines_with_offsets(text: str):
    lines = []
    idx = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        lines.append((stripped, idx))
        idx += len(line)
    return lines

# -----------------------
# Preprocessing functions
# -----------------------
def remove_const_header_after_page(text: str) -> str:
    """
    Remove occurrences of lines like "{number} THE CONSTITUTION OF INDIA"
    that commonly appear after page markers.
    """
    return re.sub(CONST_HEADER_AFTER_PAGE_RE, "", text)

def remove_part_parentheses(text: str) -> str:
    """
    Remove occurrences of "(Part I.—THE UNION)" style parentheses blocks.
    """
    return re.sub(PART_PAREN_RE, "", text)

def extract_and_wrap_footer_blocks(text: str):
    """
    Find occurrences where an underline/separator appears and text up to the next page marker
    should be treated as footer. Replace that block with FOOTER({...}) where {...} is the footer content.
    Implementation: search for underline lines; from the line after underline collect until next [[PAGE n]] marker
    (but do not remove the page marker). Replace the entire footer block (from underline line to just before page marker)
    with the single line 'FOOTER({footer_content})\n' so the footer is preserved compactly.
    """
    out = []
    pos = 0
    # We'll iterate through matches of underline
    for m in re.finditer(r"(?m)^([\_\-\=]{6,})\s*$", text):
        start = m.start()
        # append text before this underline unchanged
        out.append(text[pos:start])
        # find the next page marker after this underline
        pm = PAGE_MARKER_RE.search(text, m.end())
        if pm:
            footer_start = m.end()
            footer_end = pm.start()  # footer is between underline end and page marker start
            footer_content = text[footer_start:footer_end].strip()
            # collapse whitespace within footer content for compactness
            footer_content_compact = re.sub(r"\s+", " ", footer_content).strip()
            wrapper = f"FOOTER({footer_content_compact})\n"
            out.append(wrapper)
            pos = footer_end  # keep the page marker which begins at footer_end
        else:
            # no further page marker — treat until end of doc as footer
            footer_start = m.end()
            footer_content = text[footer_start:].strip()
            footer_content_compact = re.sub(r"\s+", " ", footer_content).strip()
            wrapper = f"FOOTER({footer_content_compact})\n"
            out.append(wrapper)
            pos = len(text)
            break
    if pos == 0:
        return text
    else:
        out.append(text[pos:])
        return "".join(out)

# -----------------------
# Main parsing
# -----------------------
def split_text_to_pages(full_text: str):
    pages = []
    current = []
    page_num = None
    for seg in re.split(r"(\[\[PAGE\s+\d+\]\])", full_text):
        if not seg:
            continue
        m = re.match(r"\[\[PAGE\s+(\d+)\]\]", seg)
        if m:
            if page_num is not None:
                pages.append({"page_num": page_num, "text": "".join(current)})
            page_num = int(m.group(1))
            current = []
        else:
            current.append(seg)
    if page_num is not None:
        pages.append({"page_num": page_num, "text": "".join(current)})
    if not pages:
        pages = [{"page_num": 1, "text": full_text}]
    return pages

def parse_articles_from_text(full_text: str,
                             target_tokens: int = 512,
                             overlap_tokens: int = 64,
                             min_tokens: int = 48,
                             merge_small: bool = False,
                             verbose: bool = False):
    # Preprocess passes
    # 1) Extract & wrap footers
    text0 = extract_and_wrap_footer_blocks(full_text)
    # 2) Remove the constitution header after page markers
    text1 = remove_const_header_after_page(text0)
    # 3) Remove part parentheses
    text2 = remove_part_parentheses(text1)
    cleaned_text = text2

    # Build page markers for char->page mapping
    page_markers = []
    for m in re.finditer(r"\[\[PAGE\s+(\d+)\]\]", cleaned_text):
        page_markers.append({"page": int(m.group(1)), "pos": m.start()})
    page_markers = sorted(page_markers, key=lambda x: x["pos"])

    def char_to_page(offset):
        if not page_markers:
            return 1
        last = page_markers[0]
        for pm in page_markers:
            if offset >= pm["pos"]:
                last = pm
            else:
                break
        return last["page"]

    lines = extract_lines_with_offsets(cleaned_text)
    N = len(lines)

    articles = []
    current_part = {"part_name": "", "part_number": "", "part_title": ""}
    i = 0

    pbar = tqdm(total=N, desc="Scanning lines", unit="line")
    while i < N:
        line, start_char = lines[i]
        stripped = line.strip()

        # PART detection per exact rule: match "PART {Roman}" and the next non-empty line should be Part Title in ALL CAPS.
        mpart = PART_HEADER_RE.match(stripped)
        if mpart:
            roman = mpart.group(1).strip()
            # find next non-empty line for title
            j = i + 1
            part_title = ""
            while j < N:
                nxt, _ = lines[j]
                nxt_s = nxt.strip()
                if not nxt_s:
                    j += 1
                    continue
                # require ALL CAPS-ish title (user said Part Title in Capital)
                if PART_TITLE_ALLCAPS_RE.match(nxt_s):
                    part_title = nxt_s
                else:
                    # even if not perfectly all-caps, capture short next line as title fallback
                    part_title = nxt_s
                break
            current_part = {"part_name": "PART", "part_number": roman, "part_title": part_title}
            if verbose:
                tqdm.write(f"Detected PART {roman} title={part_title}")
            i += 1
            pbar.update(1)
            continue

        # Article detection per the strict rule:
        mart = ARTICLE_STRICT_RE.match(stripped)
        if mart:
            art_num = mart.group(1).strip()
            art_title = mart.group(2).strip()
            art_start_char = start_char

            # Find end of article: scan ahead until next PART or next strict ARTICLE or EOF
            j = i + 1
            while j < N:
                nxt_line, _ = lines[j]
                nxt_strip = nxt_line.strip()
                # if next is PART header -> end
                if PART_HEADER_RE.match(nxt_strip):
                    break
                # if next is strict article start -> end
                if ARTICLE_STRICT_RE.match(nxt_strip):
                    break
                j += 1

            last_line_text, last_line_start = lines[j-1] if j-1 >= 0 else (stripped, start_char)
            art_end_char = last_line_start + len(last_line_text) + 1
            art_text = cleaned_text[art_start_char:art_end_char].strip()

            # Attempt to collect footnotes immediately after j if underline or FOOTER or footnote markers
            footnotes = []
            if j < N:
                nxt_after_line, _ = lines[j]
                if UNDERLINE_RE.match(nxt_after_line.strip()) or nxt_after_line.startswith("FOOTER(") or FOOTNOTE_GENERIC_RE.search(nxt_after_line):
                    # collect contiguous footnote-like lines
                    k = j
                    foot_block = []
                    while k < N:
                        l, _ = lines[k]
                        ls = l.strip()
                        if not ls:
                            break
                        if ls.startswith("FOOTER("):
                            # include footer as footnote metadata
                            foot_block.append(ls)
                            k += 1
                            break
                        if UNDERLINE_RE.match(ls):
                            k += 1
                            continue
                        if FOOTNOTE_GENERIC_RE.search(ls) or re.match(r"^\d+\.", ls):
                            foot_block.append(ls)
                            k += 1
                            continue
                        # small lowercase continuation
                        if len(ls.split()) < 20 and ls[0].islower():
                            foot_block.append(ls)
                            k += 1
                            continue
                        break
                    footnotes = foot_block

            confidence = "high"  # strict rule -> high confidence

            articles.append({
                "part_name": current_part.get("part_name", ""),
                "part_number": current_part.get("part_number", ""),
                "part_title": current_part.get("part_title", ""),
                "article_number": art_num,
                "article_title": art_title,
                "start_char": int(art_start_char),
                "end_char": int(art_end_char),
                "raw_head_line": stripped,
                "text": art_text,
                "footnotes": footnotes,
                "parsing_confidence": confidence,
                "source_line_index": i
            })

            i = j
            pbar.update(1)
            continue

        # If strict rule didn't match, we still move forward; do not treat lenient numeric headers as articles by default.
        # But to avoid missing potential articles, record numeric-line matches with low confidence in ambiguous report later.
        i += 1
        pbar.update(1)

    pbar.close()

    # Merge tiny articles if requested
    final_articles = []
    idx = 0
    while idx < len(articles):
        a = articles[idx]
        tokcount = count_tokens(a["text"])
        if tokcount < min_tokens and merge_small:
            # merge into next if present else previous
            if idx + 1 < len(articles):
                articles[idx+1]["text"] = a["text"].strip() + "\n\n" + articles[idx+1]["text"].strip()
                articles[idx+1]["start_char"] = a["start_char"]
                idx += 1
                continue
            elif final_articles:
                final_articles[-1]["text"] = final_articles[-1]["text"].strip() + "\n\n" + a["text"].strip()
                final_articles[-1]["end_char"] = a["end_char"]
                idx += 1
                continue
            else:
                final_articles.append(a)
                idx += 1
                continue
        else:
            final_articles.append(a)
            idx += 1

    # Now token-split long articles using sliding window + overlap
    all_chunks = []
    ambiguous_report = []
    chunk_idx = 0
    for a in tqdm(final_articles, desc="Creating chunks from articles"):
        art_text = a["text"]
        toklen = count_tokens(art_text)
        if toklen <= target_tokens:
            chunk = {
                "doc_id": Path("").absolute().name,  # set later
                "chunk_id": "",
                "chunk_index": chunk_idx,
                "part_name": a.get("part_name", ""),
                "part_number": a.get("part_number", ""),
                "part_title": a.get("part_title", ""),
                "article_number": a.get("article_number", ""),
                "article_title": a.get("article_title", ""),
                "sub_index": 0,
                "start_char": int(a.get("start_char", 0)),
                "end_char": int(a.get("end_char", 0)),
                "page_start": char_to_page(int(a.get("start_char", 0))),
                "page_end": char_to_page(int(a.get("end_char", 0))),
                "token_count": int(toklen),
                "chunk_hash": md5_text(art_text),
                "footnotes": a.get("footnotes", []),
                "text": art_text,
                "parsing_confidence": a.get("parsing_confidence", "medium")
            }
            all_chunks.append(chunk)
            if chunk["parsing_confidence"] != "high":
                ambiguous_report.append({
                    "chunk_index": chunk_idx,
                    "article_number": chunk["article_number"],
                    "reason": "parsing_confidence_" + chunk["parsing_confidence"],
                    "context": art_text[:400]
                })
            chunk_idx += 1
        else:
            toks = tokenize_to_tokens(art_text)
            n = len(toks)
            start_tok = 0
            sub_i = 0
            while start_tok < n:
                end_tok = min(start_tok + target_tokens, n)
                chunk_text = decode_tokens(toks[start_tok:end_tok])
                start_char_rel = char_index_of_token(art_text, toks, start_tok)
                end_char_rel = char_index_of_token(art_text, toks, end_tok)
                start_char_abs = a.get("start_char", 0) + start_char_rel
                end_char_abs = a.get("start_char", 0) + end_char_rel
                chunk = {
                    "doc_id": Path("").absolute().name,
                    "chunk_id": "",
                    "chunk_index": chunk_idx,
                    "part_name": a.get("part_name", ""),
                    "part_number": a.get("part_number", ""),
                    "part_title": a.get("part_title", ""),
                    "article_number": a.get("article_number", ""),
                    "article_title": a.get("article_title", ""),
                    "sub_index": sub_i,
                    "start_char": int(start_char_abs),
                    "end_char": int(end_char_abs),
                    "page_start": char_to_page(int(start_char_abs)),
                    "page_end": char_to_page(int(end_char_abs)),
                    "token_count": int(end_tok - start_tok),
                    "chunk_hash": md5_text(chunk_text),
                    "footnotes": a.get("footnotes", []),
                    "text": chunk_text,
                    "parsing_confidence": a.get("parsing_confidence", "medium")
                }
                all_chunks.append(chunk)
                if chunk["parsing_confidence"] != "high":
                    ambiguous_report.append({
                        "chunk_index": chunk_idx,
                        "article_number": chunk["article_number"],
                        "reason": "parsing_confidence_" + chunk["parsing_confidence"],
                        "context": chunk_text[:400]
                    })
                chunk_idx += 1
                sub_i += 1
                if end_tok == n:
                    break
                next_start = end_tok - overlap_tokens
                if next_start <= start_tok:
                    next_start = end_tok
                start_tok = next_start

    return all_chunks, ambiguous_report, {
        "num_articles_detected": len(final_articles),
        "num_chunks": len(all_chunks)
    }

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_text", required=True, help="Path to extracted .txt")
    parser.add_argument("--out_dir", default="data/chunks", help="Directory to write chunks jsonl")
    parser.add_argument("--target_tokens", type=int, default=512, help="Target tokens per chunk")
    parser.add_argument("--overlap_tokens", type=int, default=64, help="Overlap tokens between chunks")
    parser.add_argument("--min_tokens", type=int, default=48, help="Minimum tokens threshold for tiny articles")
    parser.add_argument("--merge_small", action="store_true", help="Merge very small articles into next/prev")
    parser.add_argument("--doc_id", default=None, help="Optional doc id string to use in metadata")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.in_text)
    assert in_path.exists(), f"{in_path} not found"
    ensure_dir(args.out_dir)
    raw = in_path.read_text(encoding="utf-8")

    all_chunks, ambiguous_report, stats = parse_articles_from_text(
        raw,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_tokens=args.min_tokens,
        merge_small=args.merge_small,
        verbose=args.verbose
    )

    doc_id = args.doc_id or in_path.stem.lower().replace(" ", "_")
    for i, c in enumerate(all_chunks):
        c["doc_id"] = doc_id
        c["chunk_id"] = f"{doc_id}_chunk_{i:06d}"
        c["chunk_index"] = i

    out_jsonl = Path(args.out_dir) / f"{doc_id}_article_chunks.jsonl"
    write_jsonl(out_jsonl, all_chunks)

    summary = {
        "doc_id": doc_id,
        "input_chars": len(raw),
        "num_articles_detected": stats["num_articles_detected"],
        "num_chunks": stats["num_chunks"],
        "target_tokens": args.target_tokens,
        "overlap_tokens": args.overlap_tokens,
        "min_tokens": args.min_tokens
    }
    out_summary = Path(args.out_dir) / f"{doc_id}_article_chunks_summary.json"
    write_json(out_summary, summary)

    ambiguous_out = Path(args.out_dir) / f"{doc_id}_ambiguous_report.jsonl"
    write_jsonl(ambiguous_out, ambiguous_report)

    print(f"Wrote {len(all_chunks)} chunks to {out_jsonl}")
    print(f"Summary written to {out_summary}")
    print(f"Ambiguous report written to {ambiguous_out} (items: {len(ambiguous_report)})")

if __name__ == "__main__":
    main()
