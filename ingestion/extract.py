"""Stage 1a: extract text from PDFs with Docling.

Reads data/manifest.csv, converts each PDF in data/raw/,
and writes one JSON file per document to data/processed/.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "manifest.csv"


def file_hash(path: Path) -> str:
    """Fingerprint of the file's bytes, used to skip unchanged files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


MIN_CHARS = 300  # below this, assume the text is in images and retry with OCR


def make_converter(ocr: bool = False) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = ocr                # most JU PDFs already contain real text
    if ocr:
        opts.ocr_options.mode = OcrMode.FULL_PAGE  # read the whole page as an image
    opts.do_table_structure = True   # keep tables as rows and columns
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def extract(converter: DocumentConverter, pdf: Path) -> list[dict]:
    """Return a list of elements: page, type, text."""
    elements = elements_from(converter, pdf)
    if sum(len(e["text"]) for e in elements) < MIN_CHARS:
        print(f"        {pdf.name}: little text found, retrying with OCR...")
        elements = elements_from(make_converter(ocr=True), pdf)
    return elements


def elements_from(converter: DocumentConverter, pdf: Path) -> list[dict]:
    doc = converter.convert(pdf).document
    elements = []
    for item, level in doc.iterate_items():
        label = getattr(item.label, "value", str(item.label))
        if label == "table":
            text = item.export_to_markdown(doc=doc)
        else:
            text = getattr(item, "text", "") or ""
        text = text.strip()
        if not text or not item.prov:
            continue
        elements.append({
            "page": item.prov[0].page_no,
            "type": label,
            "level": level,
            "text": text,
        })
    return elements


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    force = "--force" in args                      # re-extract even if unchanged
    only = {a for a in args if a != "--force"}    # optional: only these files

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    converter = make_converter()
    done = skipped = failed = 0

    for row in rows:
        name = row["file"]
        if only and name not in only:
            continue
        pdf = RAW / name
        out_path = OUT / (pdf.stem + ".json")
        if not pdf.exists():
            print(f"MISSING {name}: listed in manifest but not in data/raw/")
            failed += 1
            continue
        h = file_hash(pdf)

        if not force and out_path.exists() and json.loads(out_path.read_text())["sha256"] == h:
            skipped += 1
            continue

        try:
            elements = extract(converter, pdf)
        except Exception as e:
            print(f"FAILED  {name}: {e}")
            failed += 1
            continue

        record = {**row, "sha256": h, "num_elements": len(elements), "elements": elements}
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        pages = max((e["page"] for e in elements), default=0)
        print(f"OK      {name}: {len(elements)} elements, {pages} pages")
        done += 1

    print(f"\nDone: {done} extracted, {skipped} unchanged, {failed} failed")


if __name__ == "__main__":
    main()