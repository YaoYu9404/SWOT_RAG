"""
swot_rag_downloader.py
──────────────────────
Downloads SWOT publications from the NASA SWOT website and fetches
PDFs (via Unpaywall open-access API) ready to ingest into SWOT_RAG.

Pipeline:
  1. Scrape recent-publications and all-publications pages → extract DOIs
  2. Resolve each DOI → metadata (title, authors, journal, year, abstract)
  3. Fetch open-access PDF via Unpaywall (free, no key needed for basic use)
  4. Save PDFs + a JSON catalog → ready for FAISS ingestion

Usage:
  pip install requests beautifulsoup4 tqdm
  python swot_rag_downloader.py

  # Only recent papers (last 3 months):
  python swot_rag_downloader.py --mode recent

  # All papers + resume from previous run:
  python swot_rag_downloader.py --mode all --resume

  # Dry run (metadata only, no PDFs):
  python swot_rag_downloader.py --dry-run

Output:
  swot_papers/
  ├── catalog.json          ← full metadata for all found papers
  ├── pdfs/                 ← downloaded open-access PDFs
  │   ├── 10.1029_2024GL119357.pdf
  │   └── ...
  └── failed_dois.txt       ← DOIs where PDF was not freely available
"""

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ── Configuration ─────────────────────────────────────────────────────────────

SWOT_RECENT_URL = "https://swot.jpl.nasa.gov/science/publications/recent-publications/"
SWOT_ALL_URL    = "https://swot.jpl.nasa.gov/science/publications/all-publications/"
AGU_COLLECTION  = "https://agupubs.onlinelibrary.wiley.com/doi/toc/10.1002/(ISSN)1944-8007.NASASWOT1"

# Unpaywall: free open-access PDF finder (https://unpaywall.org)
# Set your email here — required by Unpaywall TOS (no account needed)
UNPAYWALL_EMAIL = "yaoyu.9404@gmail.com"

CROSSREF_BASE   = "https://api.crossref.org/works/"
UNPAYWALL_BASE  = "https://api.unpaywall.org/v2/"
SEMANTIC_BASE   = "https://api.semanticscholar.org/graph/v1/paper/DOI:"

OUTPUT_DIR      = Path("swot_papers")
PDF_DIR         = OUTPUT_DIR / "pdfs"
CATALOG_FILE    = OUTPUT_DIR / "catalog.json"
FAILED_FILE     = OUTPUT_DIR / "failed_dois.txt"

HEADERS = {
    "User-Agent": "SWOT-RAG-Downloader/1.0 (research; contact: yaoyu.9404@gmail.com)"
}
RATE_LIMIT_SEC  = 1.0   # polite delay between requests


# ── DOI extraction ─────────────────────────────────────────────────────────────

DOI_PATTERN = re.compile(
    r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+"
)

def extract_dois_from_html(html: str) -> List[str]:
    """Extract all DOIs from a page's HTML (links + text)."""
    soup = BeautifulSoup(html, "html.parser")
    dois = set()

    # From href attributes (doi.org links)
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if "doi.org" in href:
            m = DOI_PATTERN.search(href)
            if m:
                dois.add(m.group().rstrip("."))

    # From raw text (citation strings)
    for m in DOI_PATTERN.finditer(soup.get_text()):
        dois.add(m.group().rstrip("."))

    return sorted(dois)


def scrape_swot_page(url: str, session: requests.Session) -> List[str]:
    """Fetch a SWOT publications page and return all DOIs found."""
    try:
        r = session.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  Warning: could not fetch {url}: {e}")
        return []

    dois = extract_dois_from_html(r.text)
    print(f"  Found {len(dois)} DOIs at {url}")
    return dois


# ── Metadata resolution ────────────────────────────────────────────────────────

def fetch_crossref_metadata(doi: str, session: requests.Session) -> Optional[Dict]:
    """
    Fetch paper metadata from Crossref (free, no key required).
    Returns dict with: title, authors, journal, year, abstract, doi, url
    """
    url = CROSSREF_BASE + urllib.parse.quote(doi, safe="")
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json().get("message", {})
    except Exception:
        return None

    # Authors
    authors = []
    for a in data.get("author", []):
        given  = a.get("given", "")
        family = a.get("family", "")
        authors.append(f"{family}, {given}".strip(", "))

    # Year
    year = None
    for date_field in ("published-print", "published-online", "created"):
        dp = data.get(date_field, {}).get("date-parts", [[None]])
        if dp and dp[0][0]:
            year = dp[0][0]
            break

    # Title
    titles = data.get("title", [])
    title  = titles[0] if titles else ""

    # Journal
    journals = data.get("container-title", [])
    journal  = journals[0] if journals else data.get("publisher", "")

    # Abstract (Crossref sometimes has it, often doesn't)
    abstract = data.get("abstract", "")
    # Strip JATS XML tags if present
    abstract = re.sub(r"<[^>]+>", " ", abstract).strip()

    return {
        "doi":      doi,
        "url":      f"https://doi.org/{doi}",
        "title":    title,
        "authors":  authors,
        "journal":  journal,
        "year":     year,
        "abstract": abstract,
        "source":   "crossref",
    }


def enrich_with_semantic_scholar(meta: Dict, session: requests.Session) -> Dict:
    """
    Enrich metadata with Semantic Scholar (adds abstract if missing,
    adds citation count and open-access PDF link).
    """
    doi = meta.get("doi", "")
    url = SEMANTIC_BASE + urllib.parse.quote(doi, safe="")
    params = {
        "fields": "title,abstract,year,openAccessPdf,citationCount,externalIds"
    }
    try:
        r = session.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return meta
        data = r.json()
    except Exception:
        return meta

    if not meta.get("abstract") and data.get("abstract"):
        meta["abstract"] = data["abstract"]

    meta["citation_count"] = data.get("citationCount", 0)

    oa = data.get("openAccessPdf")
    if oa and oa.get("url"):
        meta["oa_pdf_url"] = oa["url"]

    return meta


# ── PDF download ───────────────────────────────────────────────────────────────

def safe_unlink(path: Path) -> None:
    """Remove a file if it exists — Python 3.7 compatible (no missing_ok)."""
    try:
        path.unlink()
    except OSError:
        pass


# Publishers that block direct PDF downloads even for OA articles.
# We skip their direct URLs and prefer repository mirrors instead.
BLOCKED_PUBLISHERS = (
    "onlinelibrary.wiley.com",
    "agupubs.onlinelibrary.wiley.com",
    "journals.ametsoc.org",
    "www.science.org",
    "www.nature.com",
    "link.springer.com",
    "www.tandfonline.com",
    "www.mdpi.com/pdf",         # MDPI sometimes blocks bots
)

# Repository hosts that are generally bot-friendly
PREFERRED_HOSTS = (
    "europepmc.org",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
    "essopenarchive.org",
    "eartharxiv.org",
    "zenodo.org",
    "hal.science",
    "osf.io",
    "repository",
    "eprints",
    "preprint",
)

# Browser-like headers to reduce 403s from publisher sites
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://scholar.google.com/",
}


def is_blocked_url(url: str) -> bool:
    """Return True if the URL is from a publisher known to block bots."""
    return any(h in url for h in BLOCKED_PUBLISHERS)


def is_preferred_url(url: str) -> bool:
    """Return True if the URL is from a repository that allows downloads."""
    return any(h in url.lower() for h in PREFERRED_HOSTS)


def fetch_unpaywall_pdf_url(doi: str,
                             session: requests.Session) -> Optional[str]:
    """
    Query Unpaywall for a legal open-access PDF URL.
    Prefers repository mirrors over publisher direct links to avoid 403s.
    Returns the best available PDF URL or None.
    """
    url = (f"{UNPAYWALL_BASE}{urllib.parse.quote(doi, safe='')}"
           f"?email={UNPAYWALL_EMAIL}")
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    all_locations = data.get("oa_locations", [])

    # Collect all candidate PDF URLs
    candidates = []
    for loc in all_locations:
        pdf_url = loc.get("url_for_pdf") or loc.get("url")
        if pdf_url:
            candidates.append(pdf_url)

    if not candidates:
        return None

    # Rank: preferred repositories first, blocked publishers last
    def rank(url):
        if is_preferred_url(url):
            return 0
        if is_blocked_url(url):
            return 2
        return 1

    candidates.sort(key=rank)
    return candidates[0]


def _try_download(url: str, dest_path: Path,
                  session: requests.Session,
                  use_browser_headers: bool = False) -> bool:
    """Single download attempt. Returns True on success."""
    hdrs = BROWSER_HEADERS if use_browser_headers else HEADERS
    try:
        r = session.get(url, headers=hdrs, timeout=60, stream=True,
                        allow_redirects=True)
        if r.status_code == 403:
            return False
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        chunks = []
        first  = True
        is_pdf = "pdf" in content_type.lower() or url.endswith(".pdf")

        for chunk in r.iter_content(chunk_size=8192):
            if first and not is_pdf:
                if not chunk[:4].startswith(b"%PDF"):
                    return False   # not a PDF — HTML redirect page etc.
                is_pdf = True
            first = False
            chunks.append(chunk)

        if not chunks:
            return False

        with open(dest_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)

        # Sanity-check file size
        if dest_path.stat().st_size < 10_000:
            safe_unlink(dest_path)
            return False

        return True

    except Exception:
        safe_unlink(dest_path)
        return False


def download_pdf(pdf_url: str, dest_path: Path,
                 session: requests.Session,
                 doi: str = "") -> bool:
    """
    Download a PDF with a multi-strategy fallback chain:
      1. Direct download with research bot headers
      2. Retry with browser headers (reduces 403s on some publishers)
      3. Try ESSOAr preprint if DOI looks like an AGU paper
      4. Try Europe PMC full-text link
    Returns True on success.
    """
    # Strategy 1: direct download
    if _try_download(pdf_url, dest_path, session, use_browser_headers=False):
        return True

    # Strategy 2: retry with browser-like headers (helps with Wiley, AMS etc.)
    safe_unlink(dest_path)
    if _try_download(pdf_url, dest_path, session, use_browser_headers=True):
        return True

    # Strategy 3: ESSOAr / EarthArXiv for AGU-family journals
    if doi:
        for preprint_base in (
            f"https://essopenarchive.org/doi/{urllib.parse.quote(doi, safe='')}",
            f"https://eartharxiv.org/repository/search/?q={urllib.parse.quote(doi)}",
        ):
            safe_unlink(dest_path)
            if _try_download(preprint_base, dest_path, session,
                              use_browser_headers=True):
                return True

    # Strategy 4: Europe PMC (good for geoscience OA content)
    if doi:
        epmc_url = (f"https://europepmc.org/search?query=DOI:{urllib.parse.quote(doi)}"
                    f"&format=pdf")
        safe_unlink(dest_path)
        if _try_download(epmc_url, dest_path, session, use_browser_headers=True):
            return True

    safe_unlink(dest_path)
    return False


def doi_to_filename(doi: str) -> str:
    """Convert DOI to a safe filename."""
    return re.sub(r"[^\w\-]", "_", doi) + ".pdf"


# ── Catalog management ─────────────────────────────────────────────────────────

def load_catalog(path: Path) -> Dict[str, Dict]:
    """Load existing catalog (keyed by DOI)."""
    if path.exists():
        with open(path) as f:
            entries = json.load(f)
        return {e["doi"]: e for e in entries}
    return {}


def save_catalog(catalog: Dict[str, Dict], path: Path) -> None:
    """Save catalog as JSON list sorted by year descending."""
    entries = sorted(catalog.values(),
                     key=lambda e: (e.get("year") or 0), reverse=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


# ── RAG-ready text export ──────────────────────────────────────────────────────

def export_rag_text(catalog: Dict[str, Dict], out_dir: Path) -> None:
    """
    Export each paper's metadata as a plain-text chunk for RAG ingestion.
    Creates swot_papers/rag_chunks/ with one .txt per paper.
    Useful when PDF is not available — abstract alone is enough for RAG.
    """
    chunks_dir = out_dir / "rag_chunks"
    chunks_dir.mkdir(exist_ok=True)

    for doi, meta in catalog.items():
        fname = chunks_dir / (doi_to_filename(doi).replace(".pdf", ".txt"))
        authors_str = "; ".join(meta.get("authors", []))
        pdf_status  = "PDF downloaded" if meta.get("pdf_path") else "No PDF (abstract only)"

        text = (
            f"TITLE: {meta.get('title', 'Unknown')}\n"
            f"AUTHORS: {authors_str}\n"
            f"JOURNAL: {meta.get('journal', '')}\n"
            f"YEAR: {meta.get('year', '')}\n"
            f"DOI: {doi}\n"
            f"URL: {meta.get('url', '')}\n"
            f"STATUS: {pdf_status}\n"
            f"CITATIONS: {meta.get('citation_count', 'N/A')}\n"
            f"\nABSTRACT:\n{meta.get('abstract', 'Not available.')}\n"
        )
        fname.write_text(text, encoding="utf-8")

    print(f"  RAG text chunks written to {chunks_dir}/ ({len(catalog)} files)")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(mode: str = "recent", dry_run: bool = False, resume: bool = False) -> None:

    OUTPUT_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    # ── 1. Scrape DOIs ────────────────────────────────────────────────────────
    print("\n── Step 1: Scraping SWOT publication pages ─────────────────────")
    all_dois: List[str] = []

    urls_to_scrape = [SWOT_RECENT_URL]
    if mode == "all":
        urls_to_scrape.append(SWOT_ALL_URL)

    for url in urls_to_scrape:
        print(f"Fetching: {url}")
        dois = scrape_swot_page(url, session)
        all_dois.extend(dois)
        time.sleep(RATE_LIMIT_SEC)

    # Deduplicate
    all_dois = sorted(set(all_dois))
    print(f"\nTotal unique DOIs found: {len(all_dois)}")

    # ── 2. Load existing catalog (for resume) ─────────────────────────────────
    catalog = load_catalog(CATALOG_FILE) if resume else {}
    failed_dois = []

    if resume and catalog:
        print(f"Resuming: {len(catalog)} papers already in catalog")
        all_dois = [d for d in all_dois if d not in catalog]
        print(f"New DOIs to process: {len(all_dois)}")

    # ── 3. Fetch metadata + PDFs ──────────────────────────────────────────────
    print("\n── Step 2: Fetching metadata & PDFs ────────────────────────────")

    for doi in tqdm(all_dois, desc="Processing DOIs", unit="paper"):
        time.sleep(RATE_LIMIT_SEC)

        # Metadata from Crossref
        meta = fetch_crossref_metadata(doi, session)
        if not meta:
            print(f"  ✗ Metadata not found: {doi}")
            failed_dois.append(doi)
            continue

        time.sleep(RATE_LIMIT_SEC)

        # Enrich with Semantic Scholar (abstract + citation count)
        meta = enrich_with_semantic_scholar(meta, session)

        if dry_run:
            meta["pdf_path"] = None
            catalog[doi] = meta
            print(f"  [DRY RUN] {meta['year']} | {meta['title'][:60]}...")
            continue

        # Try to get PDF URL from multiple sources
        pdf_url = meta.pop("oa_pdf_url", None)   # from Semantic Scholar

        if not pdf_url:
            time.sleep(RATE_LIMIT_SEC)
            pdf_url = fetch_unpaywall_pdf_url(doi, session)

        # Download PDF
        if pdf_url:
            pdf_path = PDF_DIR / doi_to_filename(doi)
            if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
                meta["pdf_path"] = str(pdf_path)
                tqdm.write(f"  ✓ (cached) {meta.get('year')} | {doi}")
            else:
                success = download_pdf(pdf_url, pdf_path, session, doi=doi)
                if success:
                    meta["pdf_path"] = str(pdf_path)
                    tqdm.write(f"  ✓ PDF      {meta.get('year')} | {doi}")
                else:
                    meta["pdf_path"] = None
                    failed_dois.append(doi)
                    tqdm.write(f"  ✗ PDF fail (all strategies) {meta.get('year')} | {doi}")
        else:
            meta["pdf_path"] = None
            failed_dois.append(doi)
            tqdm.write(f"  ✗ No OA PDF {meta.get('year')} | {doi}")

        catalog[doi] = meta

        # Save incrementally (safe against interruption)
        if len(catalog) % 5 == 0:
            save_catalog(catalog, CATALOG_FILE)

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    print("\n── Step 3: Saving outputs ──────────────────────────────────────")
    save_catalog(catalog, CATALOG_FILE)
    print(f"  Catalog saved → {CATALOG_FILE}  ({len(catalog)} papers)")

    export_rag_text(catalog, OUTPUT_DIR)

    if failed_dois:
        FAILED_FILE.write_text("\n".join(failed_dois))
        print(f"  Failed DOIs → {FAILED_FILE}  ({len(failed_dois)} papers)")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    n_pdf    = sum(1 for m in catalog.values() if m.get("pdf_path"))
    n_abs    = sum(1 for m in catalog.values() if m.get("abstract"))
    n_no_abs = len(catalog) - n_abs

    print(f"""
── Summary ─────────────────────────────────────────────────────
  Total papers in catalog : {len(catalog)}
  PDFs downloaded         : {n_pdf}
  Abstract only           : {n_abs - n_pdf}
  No metadata found       : {len(failed_dois)}
  RAG-ready chunks        : {n_pdf + n_abs}
──────────────────────────────────────────────────────────────────
  Output directory        : {OUTPUT_DIR.resolve()}
""")


# ── SWOT_RAG ingestion helper ──────────────────────────────────────────────────

def build_rag_documents(catalog_path: str = "swot_papers/catalog.json") -> List[Dict]:
    """
    Load the catalog and return a list of LangChain-compatible Document dicts.

    Usage in your SWOT_RAG pipeline:
        from swot_rag_downloader import build_rag_documents
        from langchain.schema import Document

        raw = build_rag_documents()
        docs = [Document(page_content=d['text'], metadata=d['metadata'])
                for d in raw]
        # Then pass docs to your FAISS vectorstore
    """
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    with open(path) as f:
        entries = json.load(f)

    documents = []
    for e in entries:
        # Try to read PDF text if available
        text = ""
        pdf_path = e.get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            try:
                import pypdf
                reader = pypdf.PdfReader(pdf_path)
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except Exception:
                pass

        # Fall back to abstract
        if not text.strip():
            text = e.get("abstract", "")

        if not text.strip():
            continue   # skip if nothing to embed

        documents.append({
            "text": text,
            "metadata": {
                "doi":     e.get("doi", ""),
                "title":   e.get("title", ""),
                "authors": "; ".join(e.get("authors", [])),
                "journal": e.get("journal", ""),
                "year":    e.get("year"),
                "url":     e.get("url", ""),
                "has_pdf": bool(pdf_path),
            }
        })

    print(f"Built {len(documents)} RAG documents from {len(entries)} catalog entries")
    return documents


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SWOT publications for RAG ingestion"
    )
    parser.add_argument(
        "--mode", choices=["recent", "all"], default="recent",
        help="'recent' = last 3 months only | 'all' = full bibliography (default: recent)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch metadata only, do not download PDFs"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing catalog, skip already-processed DOIs"
    )
    parser.add_argument(
        "--email", default=UNPAYWALL_EMAIL,
        help="Your email for Unpaywall API (required by their TOS)"
    )
    args = parser.parse_args()

    UNPAYWALL_EMAIL = args.email

    run(mode=args.mode, dry_run=args.dry_run, resume=args.resume)
