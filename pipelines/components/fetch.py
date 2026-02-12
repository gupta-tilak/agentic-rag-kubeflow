"""KFP v2 component — Fetch & normalise documents from multiple sources.

Step 1 of the RAG ingestion pipeline.  Accepts a **JSON list of URLs**
(or a directory path / GCS prefix), downloads every source, strips
boiler-plate, normalises whitespace, and emits a structured JSON-Lines
Dataset artifact for downstream chunking.

Structured output contract (one JSON object per line)::

    {
      "doc_id":       "<sha256-of-normalised-content>",
      "source":       "<original URL or file path>",
      "content_type": "text/html" | "text/markdown" | "text/plain",
      "title":        "<extracted or inferred title>",
      "text":         "<normalised plain-text body>",
      "fetched_at":   "<ISO-8601 timestamp>",
      "char_count":   1234
    }

Local testing
-------------
    from pipelines.components.fetch import fetch_documents
    fetch_documents.python_func(
        urls='["https://www.kubeflow.org/docs/"]',
        source_type="url",
        raw_documents=_FakeArtifact("/tmp/out.jsonl"),
    )
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "requests>=2.31,<3",
        "beautifulsoup4>=4.12,<5",
        "markdownify>=0.13,<1",
    ],
)
def fetch_documents(
    urls: str,
    source_type: str,
    raw_documents: dsl.Output[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics],
    glob_pattern: str = "**/*.*",
    request_headers: str = "{}",
    request_timeout: int = 60,
    max_retries: int = 3,
) -> str:
    """Download content from a list of sources, normalise, and emit JSON-Lines.

    Parameters
    ----------
    urls:
        JSON-encoded **list** of source identifiers.
        * ``"url"``       → ``["https://...", "https://..."]``
        * ``"directory"`` → ``["/mnt/data/docs"]``   (single path)
        * ``"gcs"``       → ``["gs://bucket/prefix"]``
    source_type:
        One of ``"url"``, ``"directory"``, ``"gcs"``.
    raw_documents:
        Output Dataset — one JSON object per line (see module docstring).
    metrics:
        Output Metrics artifact with fetch statistics.
    glob_pattern:
        File-matching glob (directory mode only).
    request_headers:
        JSON-encoded ``dict`` of HTTP headers (url mode only).
    request_timeout:
        Per-request timeout in seconds.
    max_retries:
        Number of retry attempts for transient HTTP errors.

    Returns
    -------
    str
        Human-readable summary.
    """
    import hashlib
    import json
    import logging
    import re
    import time
    import unicodedata
    from datetime import datetime, timezone
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("fetch_documents")

    # ── helpers ────────────────────────────────────────────────────
    def _normalise(text: str) -> str:
        """Unicode NFC, collapse whitespace, strip control chars."""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^\S\n]+", " ", text)       # collapse spaces (keep \n)
        text = re.sub(r"\n{3,}", "\n\n", text)       # max two consecutive newlines
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # ctrl chars
        return text.strip()

    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _extract_title_html(soup) -> str:
        """Best-effort title from HTML."""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""

    def _extract_title_md(text: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line.lstrip("# ").strip()
        return ""

    def _fetch_url(url: str, headers: dict) -> dict:
        """Download one URL with retries → structured record."""
        import requests
        from bs4 import BeautifulSoup

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=request_timeout)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = 2 ** attempt
                    log.warning("Retry %d/%d for %s (wait %ds): %s",
                                attempt, max_retries, url, wait, exc)
                    time.sleep(wait)
        else:
            raise RuntimeError(
                f"Failed to fetch {url} after {max_retries} attempts"
            ) from last_exc

        ctype = resp.headers.get("content-type", "")
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip boiler-plate tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "iframe"]):
            tag.decompose()

        if "markdown" in ctype or url.endswith((".md", ".mdx")):
            text = _normalise(soup.get_text(separator="\n", strip=True))
            title = _extract_title_md(text)
            content_type = "text/markdown"
        else:
            text = _normalise(soup.get_text(separator="\n", strip=True))
            title = _extract_title_html(soup)
            content_type = "text/html"

        return {
            "doc_id": _content_hash(text),
            "source": url,
            "content_type": content_type,
            "title": title,
            "text": text,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "char_count": len(text),
        }

    def _fetch_directory(dir_path: str) -> list:
        """Walk a local directory and load text files."""
        records = []
        root = Path(dir_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        # Use Path.glob which supports ** correctly
        for fpath in sorted(root.glob(glob_pattern)):
            if not fpath.is_file():
                continue
            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                log.warning("Skipping %s: %s", fpath, exc)
                continue
            text = _normalise(raw)
            if not text:
                continue
            suffix = fpath.suffix.lower()
            if suffix in (".md", ".mdx"):
                content_type = "text/markdown"
                title = _extract_title_md(text)
            elif suffix in (".html", ".htm"):
                content_type = "text/html"
                from bs4 import BeautifulSoup
                title = _extract_title_html(
                    BeautifulSoup(raw, "html.parser"))
            else:
                content_type = "text/plain"
                title = fpath.stem
            records.append({
                "doc_id": _content_hash(text),
                "source": str(fpath),
                "content_type": content_type,
                "title": title,
                "text": text,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "char_count": len(text),
            })
        return records

    # ── main logic ────────────────────────────────────────────────
    url_list = json.loads(urls) if isinstance(urls, str) else urls
    if not isinstance(url_list, list) or not url_list:
        raise ValueError(
            f"'urls' must be a non-empty JSON list, got: {urls!r}"
        )

    headers = json.loads(request_headers) if request_headers else {}

    documents: list[dict] = []
    errors: list[str] = []

    if source_type == "url":
        for url in url_list:
            try:
                documents.append(_fetch_url(url, headers))
                log.info("✓ %s (%d chars)", url, documents[-1]["char_count"])
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                log.error("✗ %s: %s", url, exc)

    elif source_type == "directory":
        for dir_path in url_list:
            try:
                documents.extend(_fetch_directory(dir_path))
            except Exception as exc:
                errors.append(f"{dir_path}: {exc}")
                log.error("✗ %s: %s", dir_path, exc)

    elif source_type == "gcs":
        raise NotImplementedError(
            "GCS support requires gcsfs — extend _fetch_gcs() here."
        )
    else:
        raise ValueError(
            f"Unsupported source_type={source_type!r}. "
            "Choose from: url, directory, gcs."
        )

    if not documents and errors:
        raise RuntimeError(
            f"All sources failed:\n" + "\n".join(errors)
        )

    # ── persist ───────────────────────────────────────────────────
    out_path = Path(raw_documents.path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for doc in documents:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # artifact metadata
    raw_documents.metadata["num_documents"] = len(documents)
    raw_documents.metadata["source_type"] = source_type
    raw_documents.metadata["total_chars"] = sum(d["char_count"] for d in documents)
    raw_documents.metadata["num_errors"] = len(errors)

    # KFP Metrics
    metrics.log_metric("documents_fetched", len(documents))
    metrics.log_metric("fetch_errors", len(errors))
    metrics.log_metric("total_chars", sum(d["char_count"] for d in documents))

    msg = (f"Fetched {len(documents)} documents "
           f"({len(errors)} errors) from {source_type}")
    log.info(msg)
    return msg
