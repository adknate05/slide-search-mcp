"""
One-time script to embed all slides using OpenAI text-embedding-3-small.
Run this once (or whenever you add new slides):
    OPENAI_API_KEY=sk-... python3 embed_slides.py

Cost: ~$0.01 for 1000 slides (essentially free).
"""

import json
import numpy as np
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "slide_index.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "slide_embeddings.npz")
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request

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


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set your API key first:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  python3 embed_slides.py")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print("Loading slide index...")
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    records = [r for r in all_records if not is_bad_slide(r)]
    print(f"Embedding {len(records)} slides (skipping title/empty pages)...")

    # Build input texts — prepend deck name for context
    texts = []
    for r in records:
        deck_short = os.path.splitext(r["deck_name"])[0].replace("-", " ").replace("_", " ")
        texts.append(f"{deck_short}: {clean_text(r['text'])}")

    # Embed in batches
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_vecs = [e.embedding for e in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_vecs)
        print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    # Normalize for fast cosine similarity via dot product
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_array = embeddings_array / norms

    np.savez_compressed(
        EMBEDDINGS_FILE,
        embeddings=embeddings_array,
        records=np.array([json.dumps(records)]),
    )

    total_tokens = sum(len(t.split()) * 1.3 for t in texts)  # rough estimate
    cost = (total_tokens / 1_000_000) * 0.02
    print(f"\nDone. Saved {len(all_embeddings)} embeddings to slide_embeddings.npz")
    print(f"Estimated cost: ~${cost:.4f}")


if __name__ == "__main__":
    main()
