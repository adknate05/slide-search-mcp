import os
import json
import fitz

SLIDES_DIR = "slides"
OUTPUT_FILE = "slide_index.json"


def extract_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    deck_name = os.path.basename(pdf_path)

    records = []

    for i, page in enumerate(doc):
        text = page.get_text("text", sort=True).strip()

        if not text:
            text = "[No extractable text found on this slide.]"

        records.append({
            "deck_name": deck_name,
            "page_number": i + 1,
            "text": text
        })

    return records


def main():
    if not os.path.exists(SLIDES_DIR):
        raise FileNotFoundError(
            f"Could not find folder named '{SLIDES_DIR}'. Rename your slide folder to 'slides'."
        )

    all_records = []

    for filename in os.listdir(SLIDES_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(SLIDES_DIR, filename)
            print(f"Reading: {filename}")
            all_records.extend(extract_pdf_pages(pdf_path))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\nIndexed {len(all_records)} slides/pages.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()