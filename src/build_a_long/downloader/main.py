import argparse
from pathlib import Path
from typing import List

import httpx
from bs4 import BeautifulSoup


LEGO_BASE = "https://www.lego.com"


def find_instruction_pdfs(set_number: str, locale: str = "en-us") -> List[str]:
    """
    Scrape the LEGO building instructions page for a set and return all PDF URLs.

    Args:
        set_number: The LEGO set number, e.g. "75419".
        locale: The locale segment used by lego.com, e.g. "en-us".

    Returns:
        List of absolute PDF URLs.
    """
    url = f"{LEGO_BASE}/{locale}/service/building-instructions/{set_number}"
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    pdfs: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if href and isinstance(href, str) and href.lower().endswith(".pdf"):
            # some links are relative, but LEGO uses absolute for CDN links; handle both
            if href.startswith("/"):
                href = f"{LEGO_BASE}{href}"
            pdfs.append(href)

    # Deduplicate while preserving order and filter for instruction manuals.
    # Example instruction manual URL: https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6392465.pdf
    seen = set()
    unique_pdfs = []
    for u in pdfs:
        if u not in seen and "product.bi.core.pdf" in u:
            seen.add(u)
            unique_pdfs.append(u)
    return unique_pdfs


def download(
    url: str, dest_dir: Path, *, overwrite: bool = False, show_progress: bool = True
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest = dest_dir / filename
    if dest.exists() and not overwrite:
        print(f"Skip (exists): {dest}")
        return dest
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        downloaded = 0
        last_pct = -1
        with open(dest, "wb") as f:
            for chunk in r.iter_raw(chunk_size=64 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                if show_progress:
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct != last_pct:
                            print(f"  {filename}: {pct}%", end="\r", flush=True)
                            last_pct = pct
        if show_progress:
            # Clear the progress line
            print(" " * 60, end="\r")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Download LEGO instruction manuals.")
    parser.add_argument("set_number", help="The LEGO set number, e.g. 75419")
    parser.add_argument(
        "--locale",
        default="en-us",
        help="LEGO locale to use, e.g. en-us, en-gb, de-de",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to store PDFs. Defaults to data/<set_number>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list found PDFs without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download PDFs even if the file already exists",
    )
    args = parser.parse_args()

    set_number = str(args.set_number).strip()
    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / set_number

    pdfs = find_instruction_pdfs(set_number, args.locale)
    if not pdfs:
        print(f"No PDFs found for set {set_number} (locale={args.locale}).")
        return 2

    print(f"Found {len(pdfs)} PDF(s) for set {set_number}:")
    for u in pdfs:
        print(f" - {u}")

    if args.dry_run:
        print("Dry run: no files downloaded.")
        return 0

    print(f"Downloading to: {out_dir}")
    for u in pdfs:
        dest = download(u, out_dir, overwrite=args.force, show_progress=True)
        size = dest.stat().st_size
        print(f"Downloaded {dest} ({size / 1_000_000:.2f} MB)")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
