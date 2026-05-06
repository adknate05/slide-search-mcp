import sys
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_FILE = "slide_index.json"


BAD_SLIDE_PATTERNS = [
    r"^\s*questions\??\s*$",
    r"^\s*thank you\s*$",
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
]


def load_index():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_bad_slide(record):
    text = clean_text(record["text"]).lower()
    page = int(record["page_number"])

    # Skip title slides and first-slide covers
    if page == 1:
        return True

    # Skip mostly empty slides
    if len(text) < 80:
        return True

    # Skip generic slides
    for pattern in BAD_SLIDE_PATTERNS:
        if re.search(pattern, text):
            return True

    return False


def search_slides(question, records, top_k=8):
    filtered_records = [r for r in records if not is_bad_slide(r)]

    documents = []
    for r in filtered_records:
        text = clean_text(r["text"])

        # Important: do NOT include deck name heavily in the searchable body.
        # Deck titles were overpowering the actual slide content.
        documents.append(text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=30000,
        sublinear_tf=True
    )

    doc_vectors = vectorizer.fit_transform(documents)
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(question_vector, doc_vectors).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for idx in ranked_indices:
        record = filtered_records[idx]
        results.append({
            "score": float(scores[idx]),
            "deck_name": record["deck_name"],
            "page_number": record["page_number"],
            "text": clean_text(record["text"])
        })

    return results


def print_results(question, results):
    print("\nQUESTION:")
    print(question)

    print("\nTOP MATCHING SLIDES:\n")

    for i, result in enumerate(results, start=1):
        preview = result["text"]
        preview = preview[:900] + "..." if len(preview) > 900 else preview

        print(f"{i}. {result['deck_name']} | Slide/Page {result['page_number']}")
        print(f"   Match score: {result['score']:.4f}")
        print(f"   Preview: {preview}")
        print()

    print("Use the highest relevant slide as your first place to check.")
    print("Do not trust the score blindly. Verify the preview and slide number.")


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your exam question here"')
        return

    question = " ".join(sys.argv[1:])
    records = load_index()
    results = search_slides(question, records)
    print_results(question, results)


if __name__ == "__main__":
    main()