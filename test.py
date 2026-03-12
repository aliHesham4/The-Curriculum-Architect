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

# Point to your Tesseract install (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load BLIP locally (downloads once, runs offline after)
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.eval()

# ── Load embedder once, share it with KeyBERT ──
print("Loading embedder + KeyBERT...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedder)

pdf_path = r"D:\GUC\The Curriculum Architect\Dataset\Math Curriculum For Children.pdf"
doc = fitz.open(pdf_path)
total_pages = len(doc)
chunk_size = 50

output_file = r"D:\GUC\The Curriculum Architect\Python Files\extracted_text.txt"
# ══════════════════════════════════════════════════════
#  Text cleaning functions (remove noise, filter lines)
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
#  Simple heuristic to identify TOC pages based on common phrases
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
#  Image extraction and description functions (local OCR + BLIP)
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
#  Embedding + Clustering functions (KeyBERT keywords) 
# ══════════════════════════════════════════════════════

def cluster_keywords(keywords, min_clusters=2):
    """
    Clusters KeyBERT keywords by semantic similarity.
    Names each cluster after its highest-scoring keyword.
    """
    if len(keywords) < 2:
        return {"Cluster 1": keywords}

    terms  = [kw for kw, score in keywords]
    scores = {kw: score for kw, score in keywords}

    # Embed all keywords
    embeddings = embedder.encode(terms, convert_to_numpy=True)

    # Build cosine similarity → distance matrix
    sim_matrix      = cosine_similarity(embeddings)
    distance_matrix = np.clip(1 - sim_matrix, 0, None)  # clip fixes float rounding negatives

    # ~3 keywords per cluster, minimum 2 clusters
    n_clusters = max(min_clusters, int(len(terms) / 3))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)

    # Group by label
    raw_clusters = {}
    for term, label in zip(terms, labels):
        raw_clusters.setdefault(label, []).append((term, scores[term]))

    # Sort members by score, name cluster after top keyword
    named_clusters = {}
    for members in raw_clusters.values():
        members_sorted = sorted(members, key=lambda x: x[1], reverse=True)
        top_keyword    = members_sorted[0][0]
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
#  MAIN LOOP — chunk system unchanged
# ══════════════════════════════════════════════════════

with open(output_file, "w", encoding="utf-8") as f:

    for chunk_start in range(0, total_pages, chunk_size):
        chunk_end  = min(chunk_start + chunk_size, total_pages)
        chunk_text = ""
        chunk_label = f"CHUNK PAGES {chunk_start + 1} - {chunk_end}"

        for page_number in range(chunk_start, chunk_end):
            page     = doc[page_number]
            raw_text = page.get_text()
            text     = clean_text(raw_text)

            if is_toc_page(raw_text):
                print(f"\n===== PAGE {page_number + 1} (TOC) =====")
                print(f" TOC page:\n {text}...")
                chunk_text += f"\n\n===== PAGE {page_number + 1} (TOC) =====\n{text}"
                continue

            if len(text.strip()) < 20:
                continue

            image_context = extract_page_images(page, page_number)
            page_content  = text + (f"\n{image_context}" if image_context else "")
            chunk_text   += f"\n\n===== PAGE {page_number + 1} =====\n{page_content}"

        if not chunk_text.strip():
            print(f"{chunk_label}: No usable content, skipping keywords.")
            continue

        # Save full chunk to file (keeps image tags for readability)
        f.write(f"\n\n===== {chunk_label} =====\n")
        f.write(chunk_text)

        # Strip image tags before passing to KeyBERT
        keybert_input = clean_for_keybert(chunk_text)

        keywords = kw_model.extract_keywords(
            keybert_input,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=15
        )

        if not keywords:
            print(f"{chunk_label}: No keywords found.")
            continue

        # Cluster and save
        clusters = cluster_keywords(keywords, min_clusters=2)
        print_and_save_clusters(clusters, chunk_label, f)

doc.close()
print(f"\nAll text saved to {output_file}")