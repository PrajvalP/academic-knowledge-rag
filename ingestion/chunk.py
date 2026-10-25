"""Stage 1b: split extracted documents into chunks for search.

Reads every JSON in data/processed/ (made by extract.py) and writes
data/processed/chunks.jsonl: one chunk per line, each with its text,
section heading, page range and the document's metadata.

Uses only the Python standard library.
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "chunks.jsonl"

MAX_WORDS = 250      # a chunk stops growing at about this size
OVERLAP_WORDS = 40   # words repeated when one long element must be cut
MIN_WORDS = 30       # smaller chunks are merged into their neighbour
SKIP_TYPES = {"checkbox_unselected", "checkbox_selected", "page_header", "page_footer"}
META_FIELDS = ["file", "title", "doc_type", "course_code", "school", "year",
               "access_level", "source_url"]


def find_valid_from(elements: list[dict]) -> str | None:
    """Syllabi print 'Valid From: Sep 01, 2025'. Return it as 2025-09-01."""
    text = " ".join(e["text"] for e in elements[:40])
    m = re.search(r"Valid From:\s*([A-Z][a-z]{2} \d{1,2}, \d{4})", text)
    if m:
        return datetime.strptime(m.group(1), "%b %d, %Y").date().isoformat()
    # some schools give a semester instead: "Valid From: Spring 2025"
    m = re.search(r"Valid From:\s*(Spring|Autumn)\s+(\d{4})", text)
    if m:
        month = "01" if m.group(1) == "Spring" else "08"
        return f"{m.group(2)}-{month}-01"
    return None


def split_long(text: str) -> list[str]:
    """Cut one oversized element into overlapping word windows."""
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    step = MAX_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), step)
            if words[i:i + MAX_WORDS]]


def sections(elements: list[dict]):
    """Group elements under the most recent heading."""
    heading, body = "Introduction", []
    for e in elements:
        if e["type"] in SKIP_TYPES:
            continue
        if e["type"] == "section_header":
            if body:
                yield heading, body
            heading, body = e["text"], []
        else:
            body.append(e)
    if body:
        yield heading, body


def chunk_section(heading: str, body: list[dict]):
    """Pack a section's elements into chunks of at most MAX_WORDS."""
    parts, pages, size = [], [], 0
    for e in body:
        # tables stay whole; other long elements are cut into windows
        pieces = [e["text"]] if e["type"] == "table" else split_long(e["text"])
        for piece in pieces:
            n = len(piece.split())
            if parts and size + n > MAX_WORDS:
                yield heading, parts, pages
                parts, pages, size = [], [], 0
            text = f"- {piece}" if e["type"] == "list_item" else piece
            parts.append(text)
            pages.append(e["page"])
            size += n
    if parts:
        yield heading, parts, pages


def merge_small(pieces: list[tuple]) -> list[tuple]:
    """Merge tiny chunks into the next one (or the previous, if last).

    A lone date, half a title, or a lead-in like "the student should be
    able to:" is useless on its own but useful next to what follows.
    """
    out = []
    carry = None
    for heading, parts, pages in pieces:
        if carry:
            c_heading, c_parts, c_pages = carry
            parts = [c_heading + ":"] + c_parts + [heading + ":"] + parts \
                if c_heading != heading else c_parts + parts
            pages = c_pages + pages
            carry = None
        if sum(len(p.split()) for p in parts) < MIN_WORDS:
            carry = (heading, parts, pages)
        else:
            out.append((heading, parts, pages))
    if carry:
        if out:
            h, parts, pages = out.pop()
            out.append((h, parts + [carry[0] + ":"] + carry[1], pages + carry[2]))
        else:
            out.append(carry)
    return out


def chunk_document(doc: dict) -> list[dict]:
    meta = {k: doc.get(k, "") for k in META_FIELDS}
    meta["valid_from"] = find_valid_from(doc["elements"])
    label = doc["title"] + (f" ({doc['course_code']})" if doc.get("course_code") else "")

    pieces = [p for heading, body in sections(doc["elements"])
              for p in chunk_section(heading, body)]

    chunks = []
    for h, parts, pages in merge_small(pieces):
        text = "\n".join(parts)
        chunks.append({
            **meta,
            "chunk_id": f"{Path(doc['file']).stem}#{len(chunks):03d}",
            "section": h,
            "page_start": min(pages),
            "page_end": max(pages),
            "text": text,
            # what gets embedded: the text plus where it comes from
            "embed_text": f"{label} | {doc['year']} | {h}\n{text}",
            "num_words": len(text.split()),
        })
    return chunks


def main():
    all_chunks = []
    for path in sorted(PROCESSED.glob("*.json")):
        all_chunks += chunk_document(json.loads(path.read_text(encoding="utf-8")))

    with OUT.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = sorted(c["num_words"] for c in all_chunks)
    docs = len({c["file"] for c in all_chunks})
    print(f"{len(all_chunks)} chunks from {docs} documents -> {OUT.relative_to(ROOT)}")
    print(f"words per chunk: median {sizes[len(sizes) // 2]}, "
          f"smallest {sizes[0]}, largest {sizes[-1]}")


if __name__ == "__main__":
    main()
