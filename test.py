import fitz
import re
import os
import json
from dotenv import load_dotenv
from groq import Groq
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
concepts_file = r"D:\GUC\The Curriculum Architect\Python Files\concepts.json"

load_dotenv()

print("Loading LLAMA API")
client = Groq(api_key=os.getenv("GROQ_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Say hello"}
    ]
)

print(response.choices[0].message.content)


# ══════════════════════════════════════════════════════
#  Text Cleaning
# ══════════════════════════════════════════════════════

def is_metadata_page(text):
    """Skip entire pages that are mostly legal/credits/licensing content."""
    keywords = [
        "copyright", "license", "creative commons",
        "all rights reserved", "open up resources",
        "illustrative mathematics", "credits",
        "acknowledgement", "photo credits", "illustration credits"
    ]
    count = sum(k in text.lower() for k in keywords)
    return count >= 3

# ══════════════════════════════════════════════════════
#  Text Cleaning
# ══════════════════════════════════════════════════════

def is_metadata_page(text):
    """Skip entire pages that are mostly legal/credits/licensing content."""
    keywords = [
        "copyright", "license", "creative commons",
        "all rights reserved", "open up resources",
        "illustrative mathematics", "credits",
        "acknowledgement", "photo credits", "illustration credits"
    ]
    count = sum(k in text.lower() for k in keywords)
    return count >= 3


def clean_text(text):

    # 1. Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 2. Remove control characters but preserve newlines
    text = re.sub(r'[\x00-\x09\x0B-\x1F\x7F]', ' ', text)

    # 3. Remove unicode junk
    text = re.sub(r'[\ufffd\u25a0\u25cf\u2022\u00b7\u2013\u2014\u00a0]', ' ', text)

    # 4. Remove dot leaders
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'(\. ){2,}', ' ', text)

    # 5. Block-level removal — nukes entire licensing/credits sections
    text = re.sub(
        r'(credits|copyright|licen[sc]e|creative commons).*?(?=\n\s*\n|$)',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 6. Remove standalone page numbers
    text = re.sub(r'^\s*[\-–]?\s*\d+\s*[\-–]?\s*$', '', text, flags=re.MULTILINE)

    # 7. Remove isolated special characters on their own line
    text = re.sub(r'^\s*[\W_]+\s*$', '', text, flags=re.MULTILINE)

    # 8. Remove header/footer noise
    text = re.sub(r'^\s*(grade\s*\d+|unit\s*\d+|lesson\s*\d+|page\s*\d+)\s*$',
                  '', text, flags=re.IGNORECASE | re.MULTILINE)

    # 9. Collapse whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 10. Line-level filtering — catches leftover licensing lines
    bad_line_patterns = [
        r'copyright', r'creative commons', r'licen[sc]ed',
        r'open up resources', r'illustrative mathematics',
        r'core knowledge', r'photo credits', r'illustration',
        r'all rights reserved', r'do not reproduce',
        r'may not be reproduced', r'for classroom use',
        r'©',
    ]

    # 11. Drop lines that are mostly non-alphabetic or contain bad patterns
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Skip bad pattern lines
        lower = stripped.lower()
        if any(re.search(p, lower) for p in bad_line_patterns):
            continue

        # Skip lines with too few real words
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

def is_good_keyword(kw):
    kw = kw.lower()

    bad_patterns = [
        r'\bstudents?\b',
        r'\bteacher\b',
        r'\bactivity\b',
        r'\blesson\b',
        r'\bwork\b',
        r'\bdiscussion\b',
        r'\bgroup\b',
        r'\bpractice\b',
        r'\bexample\b',
        r'\bproblem\b',
        r'\bpage\b'
    ]

    if len(kw.split()) > 4:
        return False

    if any(re.search(p, kw) for p in bad_patterns):
        return False

    return True


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

def filter_clusters(clusters, min_avg_score=0.3, min_size=2):
    filtered = {}

    for name, members in clusters.items():
        scores = [score for _, score in members]
        avg_score = sum(scores) / len(scores)

        if len(members) >= min_size and avg_score >= min_avg_score:
            filtered[name] = members

    return filtered

def cluster_coherence(members):
    terms = [kw for kw, _ in members]
    emb = embedder.encode(terms, convert_to_numpy=True)
    sim = cosine_similarity(emb)
    return sim.mean()

def filter_by_coherence(clusters, threshold=0.5):
    good = {}
    for name, members in clusters.items():
        if cluster_coherence(members) >= threshold:
            good[name] = members
    return good

def print_and_save_clusters(clusters, chunk_label, file_handle):
    header = f"\n===== KEYWORD CLUSTERS FOR {chunk_label} =====\n"
    print(header)
    file_handle.write(header)

    for cluster_name, members in clusters.items():
        line = f"\n  [{cluster_name.upper()}]\n"
        print(line, end="")
        file_handle.write(line)

        for keyword, score in members:
            kw_line = f"      {keyword}: {score:.3f}\n"
            print(kw_line, end="")
            file_handle.write(kw_line)

# ══════════════════════════════════════════════════════
#  LLM Concept Extraction
# ══════════════════════════════════════════════════════
def build_prompt(all_clusters_by_chunk, toc_context):
   

    # Format all clusters across all chunks
    clusters_section = ""
    for chunk_label, cluster_names in all_clusters_by_chunk.items():
        clusters_section += f"\n  {chunk_label}\n"
        for name in cluster_names:
            clusters_section += f"    - {name}\n"

    prompt = f"""

You are a curriculum analyst.
Use the table of contents if present and the topic clusters extracted from each 
section of a curriculum document. Your job is to use your relational reasoning to identify and link educational concepts
and their prerequisite relationships. Pass through all TOC context and cluster names to inform your analysis.
{toc_context}

ALL TOPIC CLUSTERS BY SECTION:
{clusters_section}

Your task:
1. Identify all distinct educational concepts.
2. For each concept, list its prerequisites — concepts a student must 
   understand BEFORE learning it.
3. IMPORTANT: Prerequisites must only come from concepts that also appear 
   in the clusters above. Do not invent external prerequisites.
4. Avoid making concept names too broad or too narrow. Use your judgment to find the right level of granularity.
5. If a concept has no prerequisites within this curriculum, set prerequisites to [].
6. Recognize students level based on  curriculum context and avoid suggesting prerequisites that are advanced for curriculum's target audience.

Return ONLY valid JSON, no explanation, no markdown:
{{
  "concepts": [
    {{
      "name": "concept name",
      "prerequisites": ["prerequisite 1", "prerequisite 2",...]
    }}
  ]
}}
"""
    return prompt


def query_llm(all_clusters_by_chunk, toc_context):

    print("\n===== SENDING ALL CLUSTERS TO LLaMA =====")
    prompt = build_prompt(all_clusters_by_chunk, toc_context)
    if len(prompt) > 20000:
     print("⚠ Prompt is large — consider reducing top_n or chunk count")
    else:
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            raw = response.choices[0].message.content.strip()

            # Strip code fences if model wraps output in ```json
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'^```\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parse error: {e}")
            print(f"  Raw response was:\n{raw}")
            return None
        except Exception as e:
            print(f"  ⚠ LLM error: {e}")
            return None
    
def save_concepts(parsed, file_handle):
    """Prints and saves the final concept/prerequisite list."""
    header = "\n\n══════════════════════════════════════════════════════\n"
    header += "  FINAL CONCEPTS AND PREREQUISITES\n"
    header += "══════════════════════════════════════════════════════\n"
    print(header)
    file_handle.write(header)

    if not parsed:
        msg = "  No concepts extracted.\n"
        print(msg)
        file_handle.write(msg)
        return

    for concept in parsed.get("concepts", []):
        name    = concept.get("name", "Unknown")
        prereqs = concept.get("prerequisites", [])
        prereq_str = ", ".join(prereqs) if prereqs else "None"
        line = f"  Concept: {name}\n  Prerequisites: {prereq_str}\n\n"
        print(line, end="")
        file_handle.write(line)
# ══════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════
 
# Run semantic chunking before processing
chunks, similarities = semantic_chunk(
    doc, total_pages,
    drop_threshold=0.3,
    min_pages=3,
    max_pages=20
)

# Print similarity report for inspection/tuning
print_similarity_report(similarities, chunks)
all_clusters_by_chunk = {} 
toc_context = ""
with open(output_file, "w", encoding="utf-8") as f:

    for chunk_index, page_numbers in enumerate(chunks):
        chunk_text = ""
        chunk_label = f"CHUNK {chunk_index + 1} (Pages {page_numbers[0] + 1}–{page_numbers[-1] + 1})"

        for page_number in page_numbers:
            page = doc[page_number]
            raw_text = page.get_text()
            text = clean_text(raw_text)

              # Skip metadata/licensing pages entirely
            if is_metadata_page(raw_text):
                print(f"  → Page {page_number + 1} skipped (metadata/credits)")
                continue

            if is_toc_page(raw_text):
                print(f"\n===== PAGE {page_number + 1} (TOC) =====")
                print(f" TOC page:\n {text}...")
                toc_context= text + "\n"
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

        # keywords = [kw for kw in keywords if is_good_keyword(kw[0])]

        if not keywords:
            print(f"{chunk_label}: No keywords found.")
            continue

        clusters = cluster_keywords(keywords, min_clusters=2)
        # clusters = filter_clusters(clusters, min_avg_score=0.3, min_size=2)
        print_and_save_clusters(clusters, chunk_label, f)


     # ── Accumulate cluster names for final LLaMA call ──
        all_clusters_by_chunk[chunk_label] = list(clusters.keys())

    # ── After ALL chunks: single LLaMA call with everything ──
    parsed = query_llm(all_clusters_by_chunk, toc_context)
    save_concepts(parsed, f)

    # ── Save concepts as JSON for DAG construction later ──
    if parsed:
        with open(concepts_file, "w", encoding="utf-8") as jf:
            json.dump(parsed, jf, indent=2)
        print(f"\n✅ Concepts saved to {concepts_file}")

doc.close()
print(f"\nAll text saved to {output_file}")