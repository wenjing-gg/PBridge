#!/usr/bin/env python3
"""Build the Pediatric Imaging + WHO prior knowledge bank."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import requests


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "pediatric_knowledge"
IMAGE_DIR = DATA_DIR / "images" / "pediatric_imaging"
PEDIATRIC_IMAGING_JSONL = DATA_DIR / "pediatric_imaging.jsonl"
WHO_JSONL = DATA_DIR / "who.jsonl"
KNOWLEDGE_JSONL = DATA_DIR / "knowledge.jsonl"
STATS_JSON = DATA_DIR / "knowledge_stats.json"
WHO_PDF = DATA_DIR / "who_pocket_book.pdf"

WP_API = "https://pediatricimaging.org/wp-json/wp/v2"
WHO_HANDLE = "https://iris.who.int/handle/10665/352485"
WHO_PDF_URL = (
    "https://iris.who.int/server/api/core/bitstreams/"
    "a484e069-51b4-4f27-933f-18d025350299/content"
)
PI_LICENSE = "CC BY-NC-SA 4.0"
WHO_LICENSE = "CC BY-NC-SA 3.0 IGO"
USER_AGENT = "PBridge academic research crawler; non-commercial use"


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def clean_html_text(raw_html: str) -> str:
    from bs4 import BeautifulSoup

    if "<" not in (raw_html or ""):
        return normalize_space(raw_html)
    soup = BeautifulSoup(raw_html or "", "html.parser")
    for node in soup(
        [
            "script",
            "style",
            "noscript",
            "form",
            "select",
            "option",
            "input",
            "button",
            "nav",
        ]
    ):
        node.decompose()
    for node in soup.find_all(
        ["br", "p", "li", "h1", "h2", "h3", "h4", "h5"]
    ):
        node.insert_before("\n")
    lines = [
        normalize_space(line)
        for line in soup.get_text("\n").splitlines()
    ]
    return "\n".join(line for line in lines if line)


def extract_image_urls(raw_html: str) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html or "", "html.parser")
    output: list[str] = []
    for image in soup.find_all("img"):
        candidates = [
            image.get("data-orig-file"),
            image.get("data-large-file"),
            image.get("src"),
        ]
        url = next(
            (
                str(value)
                for value in candidates
                if value and str(value).startswith("http")
            ),
            "",
        )
        if url:
            url = html.unescape(url).replace("&amp;", "&")
            if url not in output:
                output.append(url)
    return output


def image_filename(url: str) -> str:
    path = unquote(urlparse(url).path)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).name)
    if not name or "." not in name:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip("/")) + ".jpg"
    return name[:220]


def request_json(url: str, params: dict[str, Any] | None = None) -> Any:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def fetch_wp_collection(endpoint: str, fields: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        try:
            batch = request_json(
                f"{WP_API}/{endpoint}",
                {
                    "per_page": 100,
                    "page": page,
                    "_fields": fields,
                    "orderby": "id",
                    "order": "asc",
                },
            )
        except RuntimeError as error:
            if "400" in str(error) and page > 1:
                break
            raise
        if not batch:
            break
        items.extend(batch)
        print(f"{endpoint}: fetched {len(items)}", file=sys.stderr)
        if len(batch) < 100:
            break
        page += 1
    return items


def download_image(url: str) -> tuple[str, str | None]:
    destination = IMAGE_DIR / image_filename(url)
    if destination.exists() and destination.stat().st_size > 0:
        return url, str(destination.relative_to(ROOT))
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=90,
            stream=True,
        )
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("image/"):
            return url, None
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response.raw, stream)
        temporary.replace(destination)
        return url, str(destination.relative_to(ROOT))
    except (OSError, requests.RequestException):
        destination.with_suffix(destination.suffix + ".part").unlink(
            missing_ok=True
        )
        return url, None


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=120,
        stream=True,
    )
    response.raise_for_status()
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as stream:
        shutil.copyfileobj(response.raw, stream)
    temporary.replace(destination)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_pediatric_imaging(download_workers: int = 16) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    robots = urllib.robotparser.RobotFileParser(
        "https://pediatricimaging.org/robots.txt"
    )
    robots.read()
    for url in (
        "https://pediatricimaging.org/diseases/",
        "https://pediatricimaging.org/cases/",
        "https://pediatricimaging.org/posts/",
        f"{WP_API}/posts",
    ):
        if not robots.can_fetch(USER_AGENT, url):
            raise RuntimeError(f"robots.txt disallows collection from {url}")

    pages = fetch_wp_collection(
        "pages",
        "id,slug,link,title,parent,content",
    )
    posts = fetch_wp_collection(
        "posts",
        "id,slug,link,title,categories,content",
    )
    categories = fetch_wp_collection(
        "categories",
        "id,name,slug,link",
    )
    category_names = {
        item["id"]: normalize_space(item["name"]) for item in categories
    }

    candidates: list[dict[str, Any]] = []
    for page in pages:
        link = str(page.get("link", ""))
        if "/diseases/" not in link and "/intro/" not in link:
            continue
        title = clean_html_text(page.get("title", {}).get("rendered", ""))
        body_html = page.get("content", {}).get("rendered", "")
        body = clean_html_text(body_html)
        if len(body) >= 40:
            candidates.append(
                {
                    "kind": "disease/review",
                    "title": title,
                    "text": body,
                    "source_url": link,
                    "image_urls": extract_image_urls(body_html),
                }
            )

    for post in posts:
        title = clean_html_text(post.get("title", {}).get("rendered", ""))
        body_html = post.get("content", {}).get("rendered", "")
        body = clean_html_text(body_html)
        category = ", ".join(
            category_names.get(category_id, "")
            for category_id in post.get("categories", [])
            if category_names.get(category_id)
        )
        parts = [f"Clinical context: {title}"]
        if category:
            parts.append(f"Anatomical region: {category}")
        if body:
            parts.append(f"Imaging findings and caption: {body}")
        candidates.append(
            {
                "kind": "case",
                "title": title,
                "text": "\n".join(parts),
                "source_url": str(post.get("link", "")),
                "image_urls": extract_image_urls(body_html),
            }
        )

    image_urls = sorted(
        {url for candidate in candidates for url in candidate["image_urls"]}
    )
    local_images: dict[str, str] = {}
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, download_workers)) as pool:
        futures = [pool.submit(download_image, url) for url in image_urls]
        for index, future in enumerate(as_completed(futures), start=1):
            url, local_path = future.result()
            if local_path:
                local_images[url] = local_path
            else:
                failures += 1
            if index % 250 == 0 or index == len(futures):
                print(
                    f"images: {index}/{len(futures)}, failures={failures}",
                    file=sys.stderr,
                )

    deduplicated: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        text = normalize_space(
            f"{candidate['title']}. {candidate['text']}"
        )
        if len(text) < 40:
            continue
        key = text.casefold()
        paths = [
            local_images[url]
            for url in candidate["image_urls"]
            if url in local_images
        ]
        if key in deduplicated:
            deduplicated[key]["image_paths"] = sorted(
                set(deduplicated[key]["image_paths"]).union(paths)
            )
            continue
        deduplicated[key] = {
            "knowledge_id": "",
            "text": text,
            "knowledge_text": text,
            "image_paths": sorted(set(paths)),
            "source": f"Pediatric Imaging {candidate['kind']}",
            "source_url": candidate["source_url"],
            "license": PI_LICENSE,
        }

    items = sorted(
        deduplicated.values(),
        key=lambda item: (item["source"], item["source_url"], item["text"]),
    )
    for index, item in enumerate(items, start=1):
        item["knowledge_id"] = f"pi-{index:05d}"
    write_jsonl(PEDIATRIC_IMAGING_JSONL, items)
    return {
        "items": len(items),
        "image_text_items": sum(bool(item["image_paths"]) for item in items),
        "text_only_items": sum(not item["image_paths"] for item in items),
        "unique_images": len({p for item in items for p in item["image_paths"]}),
        "image_download_failures": failures,
    }


def split_text(text: str, max_chars: int = 1800) -> list[str]:
    paragraphs = [
        normalize_space(part)
        for part in re.split(
            r"\n\s*\n|\n(?=[A-Z][A-Za-z /-]{2,60}:?$)",
            text,
        )
        if normalize_space(part)
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        sentences = (
            re.split(r"(?<=[.!?])\s+", paragraph)
            if len(paragraph) > max_chars
            else [paragraph]
        )
        for sentence in sentences:
            if current and current_size + len(sentence) + 1 > max_chars:
                chunks.append(" ".join(current))
                current = []
                current_size = 0
            current.append(sentence)
            current_size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if len(chunk) >= 180]


def build_who() -> dict[str, Any]:
    import fitz

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_file(WHO_PDF_URL, WHO_PDF)
    document = fitz.open(WHO_PDF)
    items: list[dict[str, Any]] = []
    repeated = re.compile(
        r"^(POCKET BOOK OF PRIMARY HEALTH CARE FOR CHILDREN AND ADOLESCENTS|"
        r"Pocket book of primary health care for children and adolescents|"
        r"\d+)$",
        flags=re.IGNORECASE,
    )
    last_page = 0
    for page_index, page in enumerate(document):
        page_number = page_index + 1
        last_page = page_number
        if page_number <= 18:
            continue
        lines = [
            normalize_space(line)
            for line in page.get_text("text").splitlines()
            if normalize_space(line)
            and not repeated.match(normalize_space(line))
        ]
        page_text = "\n".join(lines)
        if page_number > 790 and re.search(r"\bIndex\b", page_text[:200]):
            break
        for chunk in split_text(page_text):
            items.append(
                {
                    "knowledge_id": f"who-{len(items) + 1:05d}",
                    "text": chunk,
                    "knowledge_text": chunk,
                    "image_paths": [],
                    "source": (
                        "WHO Pocket book of primary health care for "
                        "children and adolescents"
                    ),
                    "source_url": f"{WHO_HANDLE}#page={page_number}",
                    "license": WHO_LICENSE,
                }
            )
    document.close()
    write_jsonl(WHO_JSONL, items)
    return {
        "items": len(items),
        "pages": last_page,
        "pdf_bytes": WHO_PDF.stat().st_size,
    }


def merge_knowledge() -> dict[str, Any]:
    sources = [PEDIATRIC_IMAGING_JSONL, WHO_JSONL]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source knowledge files: {missing}")
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        for item in read_jsonl(source):
            text = normalize_space(item.get("text", ""))
            if not text:
                continue
            key = text.casefold()
            if key in merged:
                merged[key]["image_paths"] = sorted(
                    set(merged[key].get("image_paths", [])).union(
                        item.get("image_paths", [])
                    )
                )
                continue
            normalized = dict(item)
            normalized["text"] = text
            normalized["knowledge_text"] = normalize_space(
                item.get("knowledge_text", text)
            )
            normalized["image_paths"] = sorted(
                set(item.get("image_paths", []))
            )
            merged[key] = normalized
    items = list(merged.values())
    write_jsonl(KNOWLEDGE_JSONL, items)
    stats = {
        "total_items": len(items),
        "text_only_items": sum(not item["image_paths"] for item in items),
        "image_text_items": sum(bool(item["image_paths"]) for item in items),
        "unique_images": len({p for item in items for p in item["image_paths"]}),
        "pediatric_imaging_items": sum(
            str(item["source"]).startswith("Pediatric Imaging")
            for item in items
        ),
        "who_items": sum(str(item["source"]).startswith("WHO ") for item in items),
        "licenses": sorted({item["license"] for item in items}),
        "knowledge_text_field": "canonical raw text before GME token truncation",
    }
    STATS_JSON.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


def validate_knowledge() -> dict[str, Any]:
    items = read_jsonl(KNOWLEDGE_JSONL)
    ids = [str(item.get("knowledge_id", "")) for item in items]
    sources = {str(item.get("source", "")) for item in items}
    if len(ids) != len(set(ids)):
        raise RuntimeError("knowledge_id values are not unique")
    if not all(item.get("text") and item.get("knowledge_text") for item in items):
        raise RuntimeError("Every knowledge item needs text and knowledge_text")
    if not all(
        source.startswith("WHO ") or source.startswith("Pediatric Imaging")
        for source in sources
    ):
        raise RuntimeError(f"Unexpected provenance sources: {sorted(sources)}")
    return {
        "items": len(items),
        "unique_knowledge_ids": len(set(ids)),
        "sources": sorted(sources),
        "all_have_knowledge_text": True,
        "provenance_valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "pediatric-imaging",
            "who",
            "merge",
            "all",
            "validate",
        ],
    )
    parser.add_argument("--download-workers", type=int, default=16)
    args = parser.parse_args()

    results: dict[str, Any] = {}
    if args.command in {"pediatric-imaging", "all"}:
        results["pediatric_imaging"] = build_pediatric_imaging(
            args.download_workers
        )
    if args.command in {"who", "all"}:
        results["who"] = build_who()
    if args.command in {"merge", "all"}:
        results["knowledge"] = merge_knowledge()
    if args.command == "validate":
        results["validation"] = validate_knowledge()
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
