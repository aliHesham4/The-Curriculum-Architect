import fitz
import re
from PIL import Image
from keybert import KeyBERT
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import io

# Point to your Tesseract install (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load BLIP locally (downloads once, runs offline after)
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.eval()

# Initialize KeyBERT
kw_model = KeyBERT('all-MiniLM-L6-v2')

pdf_path = r"D:\GUC\The Curriculum Architect\Dataset\Math Curriculum For Children.pdf"
doc = fitz.open(pdf_path)
total_pages = len(doc)
chunk_size = 50

output_file = r"D:\GUC\The Curriculum Architect\Python Files\extracted_text.txt"


def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    text = re.sub(r'[\ufffd\u25a0\u25cf\u2022\u00b7]', ' ', text)
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'(\. ){2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(
        r'.*?(copyright|copyrighted|©|\(c\)|all rights reserved|reproduction prohibited'
        r'|unauthorized|licensed to|published by|printed in).*?\n',
        '', text, flags=re.IGNORECASE
    )
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        alpha_chars = sum(c.isalpha() for c in stripped)
        if alpha_chars >= 5 and alpha_chars / max(len(stripped), 1) > 0.4:
            cleaned_lines.append(stripped)
    return '\n'.join(cleaned_lines)


def is_toc_page(text):
    toc_signals = [
        r'\btable of contents\b',
        r'\bunit overview\b',
        r'\bsection overview\b',
    ]
    matches = sum(1 for pattern in toc_signals if re.search(pattern, text, re.IGNORECASE))
    return matches >= 1


def describe_image_locally(image_bytes):
    """
    Two-stage local image analysis:
    1. pytesseract → extract any text/numbers inside the image
    2. BLIP        → generate a natural language caption
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = []

        # Stage 1: OCR — great for math numbers, labels, equations
        ocr_text = pytesseract.image_to_string(pil_image).strip()
        ocr_text = re.sub(r'\s+', ' ', ocr_text)  # collapse whitespace
        if len(ocr_text) > 5:
            results.append(f"Text in image: {ocr_text}")

        # Stage 2: BLIP caption — describes what the image visually shows
        inputs = blip_processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_new_tokens=60)
        caption = blip_processor.decode(output[0], skip_special_tokens=True).strip()
        if caption:
            results.append(f"Visual: {caption}")

        return " | ".join(results) if results else "[Image: no content detected]"

    except Exception as e:
        return f"[Image analysis failed: {e}]"


def extract_page_images(page, page_number):
    """Extract all meaningful images from a page and describe them locally."""
    image_descriptions = []
    image_list = page.get_images(full=True)

    if not image_list:
        return ""

    for img_index, img in enumerate(image_list):
        xref = img[0]
        width = img[2]
        height = img[3]

        # Skip tiny decorative images
        if width < 100 or height < 100:
            continue

        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            print(f"  → Analyzing image {img_index + 1} on page {page_number + 1} ({width}x{height}px)...")
            description = describe_image_locally(image_bytes)
            image_descriptions.append(f"[IMAGE {img_index + 1}: {description}]")

        except Exception as e:
            image_descriptions.append(f"[IMAGE {img_index + 1}: Could not extract — {e}]")

    return "\n" + "\n".join(image_descriptions) + "\n" if image_descriptions else ""


with open(output_file, "w", encoding="utf-8") as f:

    for chunk_start in range(0, total_pages, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_pages)
        chunk_text = ""

        for page_number in range(chunk_start, chunk_end):
            page = doc[page_number]
            raw_text = page.get_text()
            text = clean_text(raw_text)

            if is_toc_page(raw_text):
                print(f"\n===== PAGE {page_number + 1} (TOC) =====")
                print(f" TOC page:\n {text}...")
                chunk_text += f"\n\n===== PAGE {page_number + 1} (TOC) =====\n{text}"
                continue

            if len(text.strip()) < 20:
                continue

            image_context = extract_page_images(page, page_number)
            page_content = text + (f"\n{image_context}" if image_context else "")
            chunk_text += f"\n\n===== PAGE {page_number + 1} =====\n{page_content}"

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