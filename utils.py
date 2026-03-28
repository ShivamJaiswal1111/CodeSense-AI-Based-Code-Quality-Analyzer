"""
CodeSense - Utilities
File handling, GitHub integration, exports, and helper functions.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse

from constants import LANGUAGE_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_CODE_LINES
from logger import get_logger

logger = get_logger(__name__)


# ─── File Handling ────────────────────────────────────────────────────────────

def read_file(path: str) -> Tuple[str, str]:
    """
    Read a source file and detect its language.

    Returns:
        (code, language)

    Raises:
        ValueError on unsupported file types.
        OSError on read errors.
    """
    p      = Path(path)
    suffix = p.suffix.lower()

    if suffix not in LANGUAGE_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{suffix}'.")

    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB.")

    code = p.read_text(encoding="utf-8", errors="replace")

    if len(code.splitlines()) > MAX_CODE_LINES:
        raise ValueError(f"File exceeds {MAX_CODE_LINES} lines.")

    return code, LANGUAGE_EXTENSIONS[suffix]


def detect_language_from_code(code: str) -> str:
    """Heuristic language detection from code content."""
    if re.search(r"#include\s*<|::\w+|std::", code):
        return "cpp"
    if re.search(r"\bpublic\s+class\b|\bimport\s+java\.|@Override", code):
        return "java"
    if re.search(r"\bdef\s+\w+\(|import\s+\w+|from\s+\w+\s+import", code):
        return "python"
    # Count keywords
    python_score = len(re.findall(r"\b(def|import|print|elif|lambda|None|True|False)\b", code))
    java_score   = len(re.findall(r"\b(public|private|protected|void|int|String|class)\b", code))
    cpp_score    = len(re.findall(r"\b(#include|cout|cin|namespace|template|nullptr)\b", code))
    scores = {"python": python_score, "java": java_score, "cpp": cpp_score}
    return max(scores, key=scores.get)


def code_hash(code: str) -> str:
    """Return SHA-256 hash of source code."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


# ─── GitHub Integration ───────────────────────────────────────────────────────

def fetch_github_file(url: str, timeout: int = 10) -> Tuple[str, str]:
    """
    Fetch a file from GitHub (raw URL or repository URL).

    Args:
        url:     GitHub URL (repo page or raw).
        timeout: Request timeout in seconds.

    Returns:
        (code, language)
    """
    raw_url = _github_to_raw_url(url)
    logger.info("Fetching GitHub file: %s", raw_url)

    try:
        req = Request(raw_url, headers={"User-Agent": "CodeSense/2.0"})
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status} from GitHub.")
            content = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise ValueError(f"GitHub returned HTTP {exc.code}: {exc.reason}")
    except URLError as exc:
        raise ValueError(f"Could not reach GitHub: {exc.reason}")

    # Detect language from URL extension
    parsed = urlparse(raw_url)
    suffix = Path(parsed.path).suffix.lower()
    language = LANGUAGE_EXTENSIONS.get(suffix, detect_language_from_code(content))

    return content, language


def _github_to_raw_url(url: str) -> str:
    """Convert a github.com URL to raw.githubusercontent.com."""
    if "raw.githubusercontent.com" in url:
        return url
    # github.com/user/repo/blob/branch/path → raw.githubusercontent.com/user/repo/branch/path
    m = re.match(
        r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", url
    )
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(4)}"
    raise ValueError(f"Cannot convert URL to raw: {url}")


# ─── Timing ──────────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)


# ─── Export Helpers ───────────────────────────────────────────────────────────

def results_to_markdown(results: Dict[str, Any]) -> str:
    """Convert analysis results to a readable Markdown report."""
    score   = results.get("score", 0)
    grade   = results.get("grade", "?")
    lang    = results.get("language", "unknown")
    fb      = results.get("feedback", {})

    lines = [
        f"# CodeSense Analysis Report",
        f"",
        f"**Score:** {score}/100  |  **Grade:** {grade}  |  **Language:** {lang.upper()}",
        f"",
        f"## Summary",
        f"{fb.get('opening', '')}",
        f"",
    ]

    strengths = fb.get("strengths", [])
    if strengths:
        lines += ["## ✅ Strengths", ""]
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    items = fb.get("items", [])
    errors   = [i for i in items if i["severity"] == "error"]
    warnings = [i for i in items if i["severity"] == "warning"]

    if errors:
        lines += ["## ❌ Errors", ""]
        for i in errors:
            lines.append(f"### {i['title']}")
            lines.append(i["message"])
            if i.get("code_before"):
                lines += ["```", i["code_before"], "```"]
            if i.get("code_after"):
                lines += ["**Fix:**", "```", i["code_after"], "```"]
            lines.append("")

    if warnings:
        lines += ["## ⚠️ Warnings", ""]
        for i in warnings:
            lines.append(f"### {i['title']}")
            lines.append(i["message"])
            lines.append("")

    next_steps = fb.get("next_steps", [])
    if next_steps:
        lines += ["## 📋 Next Steps", ""]
        for step in next_steps:
            lines.append(f"- {step}")

    learning = fb.get("learning_path", [])
    if learning:
        lines += ["", "## 📚 Learning Path", ""]
        for item in learning:
            r = item.get("resource", {})
            url = r.get("url", "#") if r else "#"
            lines.append(f"{item['step']}. **{item['title']}** — {item['description']} [{r.get('title','')}]({url})")

    return "\n".join(lines)


def results_to_json(results: Dict[str, Any], indent: int = 2) -> str:
    """Serialise results to pretty-printed JSON."""
    return json.dumps(results, indent=indent, default=str)


# ─── Formatting ──────────────────────────────────────────────────────────────

def truncate(text: str, max_len: int = 100) -> str:
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def format_duration(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms/1000:.1f}s"


def percentage(value: float, total: float) -> str:
    if total == 0:
        return "0%"
    return f"{(value / total * 100):.1f}%"