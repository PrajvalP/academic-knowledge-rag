"""Harvest English-taught syllabi from JU's catalogues: courses or programmes.

For each year, reads the catalogue (taught in English), downloads each
occasion's English syllabus PDF, skips exact duplicates by SHA-256 (many
occasions share one syllabus), and appends new rows to data/manifest.csv.

Polite and resumable:
  - one download per DELAY seconds, with a User-Agent naming the project
  - progress is saved in a state file in data/raw/, so a stopped run
    continues where it left off

Usage:
  python ingestion/harvest_courses.py --dry-run                 # count only
  python ingestion/harvest_courses.py                           # courses 2024-2026
  python ingestion/harvest_courses.py --programmes              # programme syllabi
  python ingestion/harvest_courses.py --programmes --dry-run 2026
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"

YEARS = [2024, 2025, 2026]   # oldest first, so a file is named after its first year
DELAY = 1.0                  # seconds between downloads
HEADERS = {"User-Agent": "academic-knowledge-rag student project (syllabus harvester)"}
FIELDS = ["file", "title", "doc_type", "course_code", "school", "year",
          "access_level", "source_url"]

# The two catalogues differ only in these settings
KINDS = {
    "courses": {
        "list_url": "https://ju.se/student/en/studies/courses.html",
        "params": {"language": "eng", "creditsMin": 0, "creditsMax": 200},
        "occasion": "CourseOccasion",
        "code_label": "Code",
        "doc_type": "syllabus",
        "file_name": "{code}_syllabus_{year}",
        "state": "harvest_state.json",
    },
    "programmes": {
        # programme syllabi are only listed on the Swedish student pages;
        # education=hp keeps university programmes (not vocational "yhp" ones)
        "list_url": "https://ju.se/student/studier/utbildningsplaner-med-kursplaner.html",
        "params": {"language": "eng", "education": "hp"},
        "occasion": "ProgrammeOccasion",
        "code_label": "Kod",
        "doc_type": "programme",
        "file_name": "programme_{code}_{year}",
        "state": "harvest_programmes_state.json",
    },
}

# Swedish school names (programme pages) -> the English names used in the manifest
SCHOOL_NAMES = {
    "Tekniska högskolan": "School of Engineering",
    "Hälsohögskolan": "School of Health and Welfare",
    "Högskolan för lärande och kommunikation": "School of Education and Communication",
    "Jönköping International Business School": "Jönköping International Business School",
    "Jönköping University Enterprise": "Jönköping University Enterprise",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def school_of(badges: list[str]) -> str:
    for b in badges:
        if b in SCHOOL_NAMES:
            return SCHOOL_NAMES[b]
        if b.startswith("School of"):
            return b
    return ""


def list_occasions(session: requests.Session, kind: dict, year: int) -> list[dict]:
    """All English-taught occasions for one year that have an English PDF."""
    params = {"take": 1000, "year": year, **kind["params"]}
    r = session.get(kind["list_url"], params=params, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    occasions = []
    for li in soup.find_all("li"):
        pdf = li.find("a", href=lambda h: h and kind["occasion"] in h and "lang=en" in h
                      and "pdf=true" in h)
        # the code span is split by invisible HTML comments ("Code<!-- -->: <!-- -->TEXV23"),
        # so match on get_text(), which joins the pieces and skips comments
        code_span = next((s for s in li.find_all("span")
                          if s.get_text("", strip=True).startswith(kind["code_label"])), None)
        if not pdf or not code_span:
            continue                       # no English PDF, or not an occasion entry
        code = code_span.get_text("", strip=True).split(":", 1)[1].strip()
        badges = [b.get_text("", strip=True) for b in li.select("span.env-badge")]
        occasions.append({
            "id": pdf["href"].split(f"{kind['occasion']}/")[1].split("?")[0],
            "title": li.find("a").get_text(strip=True),
            "course_code": code,
            "school": school_of(badges),
            "year": year,
            "url": pdf["href"],
        })

    # safety check: the page says how many it shows ("Showing"/"Visar"); if we
    # parsed none, the page layout has probably changed, so stop instead of reporting 0
    shown = soup.find(string=lambda t: t and ("Showing" in t or "Visar" in t))
    if not occasions and shown:
        raise RuntimeError(f"{year}: page says '{shown.strip()}' but none were parsed; "
                           "the page layout may have changed")
    return occasions


def main():
    # write each line immediately, even when output goes to a log file
    # (otherwise Python buffers it and `tail -f` shows nothing for a long time)
    sys.stdout.reconfigure(line_buffering=True)
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    kind = KINDS["programmes" if "--programmes" in args else "courses"]
    years = [int(a) for a in args if a.isdigit()] or YEARS
    state_file = RAW / kind["state"]

    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    # hashes of every PDF we already have, so re-harvested copies are skipped
    known = {sha256(p.read_bytes()): p.name for p in RAW.glob("*.pdf")}
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        in_manifest = {row["file"] for row in csv.DictReader(f)}

    session = requests.Session()
    session.headers.update(HEADERS)
    new_rows, stats = [], {"listed": 0, "already_done": 0, "duplicate": 0,
                           "new": 0, "failed": 0}

    for year in years:
        occasions = list_occasions(session, kind, year)
        print(f"{year}: {len(occasions)} English-taught {kind['doc_type']} occasions "
              "with an English PDF")
        stats["listed"] += len(occasions)
        if dry_run:
            continue

        for i, occ in enumerate(occasions, start=1):
            if occ["id"] in state:
                stats["already_done"] += 1
                continue
            time.sleep(DELAY)
            try:
                r = session.get(occ["url"], timeout=60)
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    raise ValueError("response is not a PDF")
            except Exception as e:
                print(f"  FAILED {occ['course_code']} {year}: {e}")
                stats["failed"] += 1
                continue          # not saved in state, so the next run retries it

            h = sha256(r.content)
            if h in known:
                state[occ["id"]] = {"duplicate_of": known[h]}
                stats["duplicate"] += 1
            else:
                base = kind["file_name"].format(code=occ["course_code"].lower(), year=year)
                name, n = f"{base}.pdf", 2
                while (RAW / name).exists():      # same code+year, different content
                    name, n = f"{base}_{n}.pdf", n + 1
                (RAW / name).write_bytes(r.content)
                known[h] = name
                state[occ["id"]] = {"file": name}
                stats["new"] += 1
                if name not in in_manifest:
                    new_rows.append({
                        "file": name, "title": occ["title"], "doc_type": kind["doc_type"],
                        "course_code": occ["course_code"], "school": occ["school"],
                        "year": year, "access_level": "public", "source_url": occ["url"],
                    })
                    in_manifest.add(name)

            if i % 25 == 0:                       # save progress regularly
                state_file.write_text(json.dumps(state, indent=1))
                print(f"  {year}: {i}/{len(occasions)} "
                      f"(new {stats['new']}, duplicates {stats['duplicate']})")
        state_file.write_text(json.dumps(state, indent=1))

    if new_rows:
        with MANIFEST.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerows(new_rows)   # quotes commas

    print(f"\nDone{' (dry run)' if dry_run else ''}: {stats}")
    if new_rows:
        print(f"Added {len(new_rows)} rows to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
