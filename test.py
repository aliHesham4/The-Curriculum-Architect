import fitz
import re
from keybert import KeyBERT

# Initialize KeyBERT
kw_model = KeyBERT('all-MiniLM-L6-v2')

# PDF path
pdf_path = r"D:\GUC\The Curriculum Architect\Dataset\Math Curriculum For Children.pdf"
doc = fitz.open(pdf_path)
total_pages = len(doc)
chunk_size = 50

output_file = r"D:\GUC\The Curriculum Architect\Python Files\extracted_text.txt"


def clean_text(text):
    # 1. Remove control characters (backspace, form feed, null, etc.)
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)

    # 2. Remove unicode replacement characters and other junk symbols
    text = re.sub(r'[\ufffd\u25a0\u25cf\u2022\u00b7]', ' ', text)

    # 3. Remove dot leaders: "Chapter Name . . . . . . 12" or "Chapter......12"
    text = re.sub(r'\.{2,}', ' ', text)           # "......" style
    text = re.sub(r'(\. ){2,}', ' ', text)        # ". . . . ." style

    # 4. Remove standalone page numbers left after dot leader removal (e.g., lone digits on a line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 5. Remove copyright lines (case-insensitive)
    text = re.sub(
        r'.*?(copyright|copyrighted|©|\(c\)|all rights reserved|reproduction prohibited'
        r'|unauthorized|licensed to|published by|printed in).*?\n',
        '', text, flags=re.IGNORECASE
    )

   
    # 6. Remove lines that are mostly non-alphabetic (leftover symbols, page refs, etc.)
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        alpha_chars = sum(c.isalpha() for c in stripped)
        # Keep line only if it has enough real words
        if alpha_chars >= 5 and alpha_chars / max(len(stripped), 1) > 0.4:
            cleaned_lines.append(stripped)

    return '\n'.join(cleaned_lines)


def is_toc_page(text):
    """Detect if a page is primarily a Table of Contents or boilerplate."""
    toc_signals = [
        r'\btable of contents\b',      # if this alone is found → skip
        r'\bunit overview\b',
        r'\bsection overview\b',
    ]
    matches = sum(1 for pattern in toc_signals if re.search(pattern, text, re.IGNORECASE))
    return matches >= 1 


with open(output_file, "w", encoding="utf-8") as f:

    for chunk_start in range(0, total_pages, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_pages)
        chunk_text = ""

        for page_number in range(chunk_start, chunk_end): # Loop through pages in the current chunk
            page = doc[page_number]
            raw_text = page.get_text()
            text = clean_text(raw_text)
            
            if is_toc_page(raw_text):
                print(f"\n===== PAGE {page_number + 1} (TOC) =====")
                print(f" TOC page:\n {text}...")  # Print a snippet of the TOC page for verification
                chunk_text += f"\n\n===== PAGE {page_number + 1} (TOC) =====\n{text}"
                continue

            

            if len(text.strip()) < 20:
                continue

            chunk_text += f"\n\n===== PAGE {page_number + 1} =====\n{text}"

        if not chunk_text.strip():
            print(f"Chunk {chunk_start + 1}-{chunk_end}: No usable content, skipping keywords.")
            continue

        f.write(f"\n\n===== CHUNK PAGES {chunk_start + 1} - {chunk_end} =====\n")
        f.write(chunk_text)

        keywords = kw_model.extract_keywords(
            chunk_text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=15
        )

        print(f"\n\n===== KEYWORDS FOR CHUNK {chunk_start + 1} - {chunk_end} =====")
        for kw, score in keywords:
            print(f"{kw}: {score:.3f}")

doc.close()
print(f"\nAll text saved to {output_file}")