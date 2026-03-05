import fitz
import re
from keybert import KeyBERT

kw_model = KeyBERT('all-MiniLM-L6-v2')

# Open PDF
pdf_path = r"D:\GUC\The Curriculum Architect\Test\Math Curriculum For Children.pdf"
doc = fitz.open(pdf_path)
total_pages = len(doc)
chunk_size = 50  # pages per chunk

# Output file
output_file = r"D:\GUC\The Curriculum Architect\Test\extracted_text.txt"

# Function to clean PDF text
def clean_text(text):
    # Remove control characters like backspace, form feed, etc.
    return re.sub(r'[\x00-\x1F\x7F]', '', text)

# Open file to save text and keywords
with open(output_file, "w", encoding="utf-8") as f:

    # Loop through the PDF in chunks
    for chunk_start in range(0, total_pages, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_pages)
        chunk_text = ""

        # Process each page in the chunk
        for page_number in range(chunk_start, chunk_end):
            page = doc[page_number]
            text = clean_text(page.get_text())

            if len(text) < 20:  # skip almost empty pages
                continue

            # Detect Table of Contents / Chapter pages
            if re.search(r'\b(table of contents|contents|chapter)', text, re.IGNORECASE):
                toc_msg = f"\n=== PAGE {page_number + 1} (TOC / Chapter) ===\n{text}\n"
                print(toc_msg)
                f.write(toc_msg)

            # Append page text to chunk
            chunk_text += f"\n\n===== PAGE {page_number + 1} =====\n{text}"

        # Save chunk text to file
        f.write(f"\n\n===== CHUNK PAGES {chunk_start + 1} - {chunk_end} =====\n")
        f.write(chunk_text)

        # Extract keywords for the chunk
        keywords = kw_model.extract_keywords(
            chunk_text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=15
        )

        # Print keywords
        kw_msg = f"\n\n===== KEYWORDS FOR CHUNK {chunk_start + 1} - {chunk_end} =====\n"
        print(kw_msg)
        for kw, score in keywords:
            line = f"{kw}: {score:.3f}\n"
            print(line.strip())
            

# Close the PDF
doc.close()
