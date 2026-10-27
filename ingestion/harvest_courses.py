"""Harvest English-taught course syllabi from JU's course catalogue.

For each year, reads the course list (taught in English), downloads each
occasion's English syllabus PDF, skips exact duplicates by SHA-256 (many
occasions share one syllabus), and appends new rows to data/manifest.csv.

Polite and resumable:
  - one download per DELAY seconds, with a User-Agent naming the project
  - progress is saved in data/raw/harvest_state.json, so a stopped run
    continues where it left off

Usage:
  python ingestion/harvest_courses.py --dry-run     # count only, no downloads
  python ingestion/harvest_courses.py               # harvest 2024-2026
  python ingestion/harvest_courses.py 2026          # one year
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
STATE = RAW / "harvest_state.json"

LIST_URL = "https://ju.se/student/en/studies/courses.html"
YEARS = [2024, 2025, 2026]   # oldest first, so a file is named after its first year
DELAY = 1.0                  # seconds between downloads
HEADERS = {"User-Agent": "academic-knowledge-rag student project (syllabus harvester)"}
FIELDS = ["file", "title", "doc_type", "course_code", "school", "year",
          "access_level", "source_url"]
SCHOOLS = ("School of", "Jönköping International Business School",
           "Jönköping University Enterprise")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def list_occasions(session: requests.Session, year: int) -> list[dict]:
    """All English-taught course occasions for one year, from the catalogue."""
    params = {"take": 1000, "year": year, "language": "eng",
              "creditsMin": 0, "creditsMax": 200}
    r = session.get(LIST_URL, params=params, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    occasions = []
    for li in soup.find_all("li"):
        pdf = li.find("a", href=lambda h: h and "syllabus.ju.se" in h and "lang=en" in h
                      and "pdf=true" in h)
        # the code span is split by invisible HTML comments ("Code<!-- -->: <!-- -->TEXV23"),
        # so match on get_text(), which joins the pieces and skips comments
        code_span = next((s for s in li.find_all("span")
                          if s.get_text("", strip=True).startswith("Code")), None)
        if not pdf or not code_span:
            continue                       # no English PDF, or not a course entry
        code = code_span.get_text("", strip=True).split(":", 1)[1].strip()
        badges = [b.get_text("", strip=True) for b in li.select("span.env-badge")]
        school = next((b for b in badges if b.startswith(SCHOOLS)), "")
        occasions.append({
            "id": pdf["href"].split("CourseOccasion/")[1].split("?")[0],
            "title": li.find("a").get_text(strip=True),
            "course_code": code,
            "school": school,
            "year": year,
            "url": pdf["href"],
        })

    # safety check: the page says how many courses it shows; if we parsed none,
    # the page layout has probably changed, so stop instead of reporting 0
    shown = soup.find(string=lambda t: t and "Showing" in t and "of" in t)
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
    years = [int(a) for a in args if a.isdigit()] or YEARS

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    # hashes of every PDF we already have, so re-harvested copies are skipped
    known = {sha256(p.read_bytes()): p.name for p in RAW.glob("*.pdf")}
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        in_manifest = {row["file"] for row in csv.DictReader(f)}

    session = requests.Session()
    session.headers.update(HEADERS)
    new_rows, stats = [], {"listed": 0, "already_done": 0, "duplicate": 0,
                           "new": 0, "failed": 0}

    for year in years:
        occasions = list_occasions(session, year)
        print(f"{year}: {len(occasions)} English-taught occasions with an English syllabus")
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
                name = f"{occ['course_code'].lower()}_syllabus_{year}.pdf"
                n = 2
                while (RAW / name).exists():      # same code+year, different content
                    name = f"{occ['course_code'].lower()}_syllabus_{year}_{n}.pdf"
                    n += 1
                (RAW / name).write_bytes(r.content)
                known[h] = name
                state[occ["id"]] = {"file": name}
                stats["new"] += 1
                if name not in in_manifest:
                    new_rows.append({
                        "file": name, "title": occ["title"], "doc_type": "syllabus",
                        "course_code": occ["course_code"], "school": occ["school"],
                        "year": year, "access_level": "public", "source_url": occ["url"],
                    })
                    in_manifest.add(name)

            if i % 25 == 0:                       # save progress regularly
                STATE.write_text(json.dumps(state, indent=1))
                print(f"  {year}: {i}/{len(occasions)} "
                      f"(new {stats['new']}, duplicates {stats['duplicate']})")
        STATE.write_text(json.dumps(state, indent=1))

    if new_rows:
        with MANIFEST.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerows(new_rows)   # quotes commas

    print(f"\nDone{' (dry run)' if dry_run else ''}: {stats}")
    if new_rows:
        print(f"Added {len(new_rows)} rows to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
