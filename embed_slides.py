"""
One-time script to embed all slides using Ollama nomic-embed-text.
Run this once (or whenever you add new slides):
    python3 embed_slides.py
"""

import json
import numpy as np
import httpx
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "slide_index.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "slide_embeddings.npz")
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

BAD_SLIDE_PATTERNS = [
    r"^\s*questions?\s*$",
    r"^\s*thank you\s*$",
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
]


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_bad_slide(record: dict) -> bool:
    text = clean_text(record["text"]).lower()
    if int(record["page_number"]) == 1:
        return True
    if len(text) < 80:
        return True
    for pattern in BAD_SLIDE_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def embed_text(client: httpx.Client, text: str) -> list[float]:
    response = client.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def main():
    print("Loading slide index...")
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    records = [r for r in all_records if not is_bad_slide(r)]
    print(f"Embedding {len(records)} slides (skipping title/empty pages)...")

    embeddings = []
    indices = []  # maps back to position in all_records

    all_map = {id(r): i for i, r in enumerate(all_records)}

    with httpx.Client() as client:
        # Check Ollama is running
        try:
            client.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        except Exception:
            print("ERROR: Ollama is not running. Start it with: ollama serve")
            sys.exit(1)

        for i, record in enumerate(records):
            text = clean_text(record["text"])
            # Prepend deck name to give context to the embedding
            deck_short = os.path.splitext(record["deck_name"])[0].replace("-", " ").replace("_", " ")
            embed_input = f"{deck_short}: {text}"

            try:
                vec = embed_text(client, embed_input)
                embeddings.append(vec)
                indices.append(i)
            except Exception as e:
                print(f"  WARNING: failed to embed slide {i} ({record['deck_name']} p{record['page_number']}): {e}")
                continue

            if (i + 1) % 50 == 0 or (i + 1) == len(records):
                print(f"  {i + 1}/{len(records)}")

    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Normalize for fast cosine similarity via dot product
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_array = embeddings_array / norms

    # Save: embeddings matrix + the filtered records as JSON string
    filtered_records_json = json.dumps([records[i] for i in indices])
    np.savez_compressed(
        EMBEDDINGS_FILE,
        embeddings=embeddings_array,
        records=np.array([filtered_records_json]),
    )

    print(f"\nDone. Saved {len(embeddings)} embeddings to slide_embeddings.npz")


if __name__ == "__main__":
    main()
