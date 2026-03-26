import fitz
import re
from PIL import Image
from keybert import KeyBERT
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import io
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════
#  Setup
# ══════════════════════════════════════════════════════

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.eval()

print("Loading embedder + KeyBERT...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedder)

pdf_path = r"D:\GUC\The Curriculum Architect\Dataset\Math Curriculum For Children.pdf"
doc = fitz.open(pdf_path)
total_pages = len(doc)

output_file = r"D:\GUC\The Curriculum Architect\Python Files\extracted_text.txt"


# ══════════════════════════════════════════════════════
#  Text Cleaning
# ══════════════════════════════════════════════════════

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


def clean_for_keybert(text):
    """Strip image wrapper tags so KeyBERT only sees clean content."""
    text = re.sub(r'\[IMAGE \d+:\s*', '', text)
    text = re.sub(r'\]', '', text)
    text = re.sub(r'Text in image:\s*', '', text)
    text = re.sub(r'Visual:\s*', '', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


# ══════════════════════════════════════════════════════
#  TOC Detection
# ══════════════════════════════════════════════════════

def is_toc_page(text):
    toc_signals = [
        r'\btable of contents\b',
        r'\bunit overview\b',
        r'\bsection overview\b',
    ]
    matches = sum(1 for pattern in toc_signals if re.search(pattern, text, re.IGNORECASE))
    return matches >= 1


# ══════════════════════════════════════════════════════
#  Image Extraction
# ══════════════════════════════════════════════════════

def describe_image_locally(image_bytes):
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = []

        ocr_text = pytesseract.image_to_string(pil_image).strip()
        ocr_text = re.sub(r'\s+', ' ', ocr_text)
        if len(ocr_text) > 5:
            results.append(f"Text in image: {ocr_text}")

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
    image_descriptions = []
    image_list = page.get_images(full=True)

    if not image_list:
        return ""

    for img_index, img in enumerate(image_list):
        xref = img[0]
        width = img[2]
        height = img[3]

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


# ══════════════════════════════════════════════════════
#  Semantic Chunking
# ══════════════════════════════════════════════════════

def semantic_chunk(doc, total_pages,
                   drop_threshold,
                   min_pages,
                   max_pages):
    """
    Chunks purely by semantic topic shift between pages.
    No layout knowledge required — works on any PDF structure.

    drop_threshold: 0.0 - 1.0
        lower  = more sensitive, more chunks, catches subtle shifts
        higher = less sensitive, fewer chunks, only catches big shifts
        0.25 is a safe default — tune up if chunks are too small
    """
    print("Embedding pages for semantic chunking...")
    page_texts = []
    for i in range(total_pages):
        text = clean_text(doc[i].get_text())
        page_texts.append(text if len(text.strip()) > 20 else "empty page")

    embeddings = embedder.encode(page_texts, convert_to_numpy=True, show_progress_bar=True)

    similarities = []
    for i in range(1, total_pages):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        similarities.append(sim)

    chunks = []
    current_chunk = [0]

    for i, sim in enumerate(similarities):
        next_page = i + 1
        current_size = len(current_chunk)

        if current_size >= max_pages:
            chunks.append(current_chunk)
            print(f"  → Forced split at page {next_page + 1} (max pages reached)")
            current_chunk = [next_page]
            continue

        if sim < (1 - drop_threshold) and current_size >= min_pages:
            chunks.append(current_chunk)
            print(f"  → Semantic boundary at page {next_page + 1} (similarity={sim:.3f})")
            current_chunk = [next_page]
            continue

        current_chunk.append(next_page)

    if current_chunk:
        chunks.append(current_chunk)

    print(f"\n✅ Semantic chunking: {len(chunks)} chunks from {total_pages} pages\n")
    return chunks, similarities


def print_similarity_report(similarities, chunks):
    """Prints a visual map of where boundaries were placed."""
    print("\n===== SIMILARITY REPORT =====")
    print("(lower score = bigger topic shift between pages)\n")

    boundary_pages = set()
    for chunk in chunks:
        if chunk:
            boundary_pages.add(chunk[0])

    for i, sim in enumerate(similarities):
        page = i + 1
        bar = "█" * int(sim * 20)
        marker = " ← BOUNDARY" if (i + 1) in boundary_pages else ""
        print(f"  Page {page:>4} → {page + 1:<4} | {bar:<20} | {sim:.3f}{marker}")


# ══════════════════════════════════════════════════════
#  Embedding Clustering
# ══════════════════════════════════════════════════════

def cluster_keywords(keywords, min_clusters=2):
    """
    Clusters KeyBERT keywords by semantic similarity.
    Names each cluster after its highest-scoring keyword.
    """
    if len(keywords) < 2:
        return {"Cluster 1": keywords}

    terms = [kw for kw, score in keywords]
    scores = {kw: score for kw, score in keywords}

    embeddings = embedder.encode(terms, convert_to_numpy=True)
    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = np.clip(1 - sim_matrix, 0, None)

    n_clusters = max(min_clusters, int(len(terms) / 3))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)

    raw_clusters = {}
    for term, label in zip(terms, labels):
        raw_clusters.setdefault(label, []).append((term, scores[term]))

    named_clusters = {}
    for members in raw_clusters.values():
        members_sorted = sorted(members, key=lambda x: x[1], reverse=True)
        top_keyword = members_sorted[0][0]
        named_clusters[top_keyword] = members_sorted

    return named_clusters


def print_and_save_clusters(clusters, chunk_label, file_handle):
    header = f"\n===== KEYWORD CLUSTERS FOR {chunk_label} =====\n"
    print(header)
    file_handle.write(header)

    for cluster_name, members in clusters.items():
        line = f"\n  [{cluster_name.upper()}]\n"
        print(line, end="")
        file_handle.write(line)
        for keyword, score in members:
            entry = f"      {keyword}: {score:.3f}\n"
            print(entry, end="")
            file_handle.write(entry)


# ══════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════

# Run semantic chunking before processing
chunks, similarities = semantic_chunk(
    doc, total_pages,
    drop_threshold=0.45,
    min_pages=3,
    max_pages=20
)

# Print similarity report for inspection/tuning
print_similarity_report(similarities, chunks)

with open(output_file, "w", encoding="utf-8") as f:

    for chunk_index, page_numbers in enumerate(chunks):
        chunk_text = ""
        chunk_label = f"CHUNK {chunk_index + 1} (Pages {page_numbers[0] + 1}–{page_numbers[-1] + 1})"

        for page_number in page_numbers:
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
            print(f"{chunk_label}: No usable content, skipping.")
            continue

        f.write(f"\n\n===== {chunk_label} =====\n")
        f.write(chunk_text)

        keybert_input = clean_for_keybert(chunk_text)

        keywords = kw_model.extract_keywords(
            keybert_input,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=10
        )

        if not keywords:
            print(f"{chunk_label}: No keywords found.")
            continue

        clusters = cluster_keywords(keywords, min_clusters=2)
        print_and_save_clusters(clusters, chunk_label, f)

doc.close()
print(f"\nAll text saved to {output_file}")