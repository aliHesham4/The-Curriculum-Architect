from sklearn.metrics.pairwise import cosine_similarity
from config import embedder
from cleaning import clean_text


def semantic_chunk(doc, total_pages, drop_threshold, min_pages, max_pages):
    print("Embedding pages for semantic chunking...")
    page_texts = []
    for i in range(total_pages):
        text = clean_text(doc[i].get_text())
        page_texts.append(text if len(text.strip()) > 20 else "empty page")

    embeddings  = embedder.encode(page_texts, convert_to_numpy=True, show_progress_bar=True)
    similarities = []
    for i in range(1, total_pages):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        similarities.append(sim)

    chunks        = []
    current_chunk = [0]

    for i, sim in enumerate(similarities):
        next_page    = i + 1
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
    print("\n===== SIMILARITY REPORT =====")
    print("(lower score = bigger topic shift between pages)\n")
    boundary_pages = {chunk[0] for chunk in chunks if chunk}

    for i, sim in enumerate(similarities):
        page   = i + 1
        bar    = "█" * int(sim * 20)
        marker = " ← BOUNDARY" if (i + 1) in boundary_pages else ""
        print(f"  Page {page:>4} → {page + 1:<4} | {bar:<20} | {sim:.3f}{marker}")