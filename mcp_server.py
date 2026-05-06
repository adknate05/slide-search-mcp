"""
MCP server for semantic slide search.
Runs locally (stdio) or on Railway (SSE/HTTP) depending on environment.

Setup locally:
    1. python3 embed_slides.py        (one-time, builds slide_embeddings.npz)
    2. python3 setup_drive.py         (optional, adds clickable Google Drive links)
    3. Restart Claude desktop app

Deploy to Railway:
    - Set OPENAI_API_KEY env var in Railway dashboard
    - Railway auto-detects and runs via railway.toml
"""

import json
import os
import re

import numpy as np
from mcp.server.fastmcp import FastMCP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "slide_embeddings.npz")
DRIVE_LINKS_FILE = os.path.join(BASE_DIR, "drive_links.json")
EMBED_MODEL = "text-embedding-3-small"

mcp = FastMCP("slide-search")

# ── Load data once at startup ─────────────────────────────────────────────────

_embeddings: np.ndarray | None = None
_records: list[dict] | None = None
_drive_links: dict[str, str] = {}


def _load():
    global _embeddings, _records, _drive_links

    if _embeddings is not None:
        return

    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(
            f"{EMBEDDINGS_FILE} not found. Run: python3 embed_slides.py"
        )

    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    _embeddings = data["embeddings"]  # pre-normalized
    _records = json.loads(str(data["records"][0]))

    if os.path.exists(DRIVE_LINKS_FILE):
        with open(DRIVE_LINKS_FILE, "r", encoding="utf-8") as f:
            _drive_links = json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

_WHY_PATTERN = re.compile(
    r"^\s*why\s+(was|were|is|are|did|do|does|has|have|would|could|should)\b",
    re.IGNORECASE,
)
_HOW_CAUSE_PATTERN = re.compile(
    r"^\s*how\s+(did|does|do|was|were)\b.*\b(happen|occur|start|begin|lead|cause)\b",
    re.IGNORECASE,
)


def _rewrite_query(question: str) -> str:
    q = question.strip()
    if _WHY_PATTERN.match(q) or _HOW_CAUSE_PATTERN.match(q):
        core = re.sub(r"^\s*(why|how)\s+", "", q, flags=re.IGNORECASE).rstrip("?").strip()
        return f"reasons causes explanations motivations {core}"
    return q


def _embed_query(question: str) -> np.ndarray:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    rewritten = _rewrite_query(question)
    response = client.embeddings.create(model=EMBED_MODEL, input=rewritten)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _clean(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _drive_link(filename: str, page: int) -> str | None:
    file_id = _drive_links.get(filename)
    if not file_id:
        return None
    return f"https://drive.google.com/file/d/{file_id}/view#page={page}"


# ── MCP Tool ──────────────────────────────────────────────────────────────────


@mcp.tool()
def search_slides(question: str, top_k: int = 8) -> str:
    """
    Semantically search class slides for the most relevant pages to answer an exam question.
    Returns the top matching slides with their deck name, page number, text preview,
    and a clickable Google Drive link that opens the PDF at the exact page (if Drive is set up).

    ALWAYS call this tool before answering ANY question about course content — drugs, pharmacology,
    history, opioids, addiction, law, ethics, or any other topic that could appear in class slides.
    Never answer from general knowledge alone; always ground your answer in the slides first.

    Args:
        question: The exam question or topic to search for.
        top_k: Number of top results to return (default 8).
    """
    _load()

    query_vec = _embed_query(question)
    scores = _embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        record = _records[idx]
        text = _clean(record["text"])
        preview = text[:800] + "..." if len(text) > 800 else text
        link = _drive_link(record["deck_name"], record["page_number"])
        results.append({
            "score": round(float(scores[idx]), 4),
            "deck": record["deck_name"],
            "page": record["page_number"],
            "link": link,
            "preview": preview,
        })

    has_links = any(r["link"] for r in results)
    lines = [f"Top {top_k} slides for: {question!r}\n"]

    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['deck']} — Page {r['page']}  (score: {r['score']})")
        if r["link"]:
            lines.append(f"   🔗 {r['link']}")
        lines.append(f"   {r['preview']}")
        lines.append("")

    if not has_links:
        lines.append("💡 Tip: run `python3 setup_drive.py` to add clickable Google Drive links.")

    return "\n".join(lines)


if __name__ == "__main__":
    # Railway sets PORT; run SSE transport for remote access, stdio for local
    port = int(os.environ.get("PORT", 0))
    if port:
        mcp.run(transport="sse", host="0.0.0.0", port=port)
    else:
        mcp.run()
