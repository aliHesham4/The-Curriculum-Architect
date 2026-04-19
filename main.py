import json
from config import doc, OUTPUT_FILE, CONCEPTS_FILE,CLEAN_RELATIONS_FILE,DETERMINSITIC_VALIDATOR_OUTPUT, kw_model
from cleaning import clean_text, clean_for_keybert, is_good_keyword
from detection import is_toc_page
from images import extract_page_images
from chunking import semantic_chunk, print_similarity_report
from clustering import cluster_keywords, filter_by_coherence, print_and_save_clusters
from llm import query_llm, save_concepts, build_document_index, save_relation_scores,validate_prerequisite_ordering
from DAG import build_networkx_dag, print_dag_summary, plot_dag


pages, page_embeddings = build_document_index()

chunks, similarities = semantic_chunk(
    pages,
    drop_threshold=0.45,
    min_pages=3,
    max_pages=20
)

print_similarity_report(similarities, chunks)

all_clusters_by_chunk = {}
toc_context           = ""
toc_page_count = 0
toc_ended      = False

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for chunk_index, page_numbers in enumerate(chunks):
        chunk_text  = ""
        chunk_label = f"CHUNK {chunk_index + 1} (Pages {page_numbers[0] + 1}–{page_numbers[-1] + 1})"

        for page_number in page_numbers:
            page     = doc[page_number]
            raw_text = page.get_text()

            if not toc_ended and toc_page_count < 3 and is_toc_page(raw_text):
                toc_page_count += 1
                text = clean_text(raw_text)
                toc_context += text + "\n"
                chunk_text  += f"\n\n===== PAGE {page_number + 1} (TOC) =====\n{text}"
                continue
            else:
                if toc_page_count > 0:
                    toc_ended = True  # TOC block is done, never detect again

            text = clean_text(raw_text)
            if len(text.strip()) < 20:
                chunk_text += f"\n\n===== PAGE {page_number + 1} =====\n"
                continue

            image_context = extract_page_images(page, page_number)
            page_content  = text + (f"\n{image_context}" if image_context else "")
            chunk_text   += f"\n\n===== PAGE {page_number + 1} =====\n{page_content}"

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
            top_n=15,
            # use_maxsum=True
        )
        keywords = [kw for kw in keywords if is_good_keyword(kw[0])]

        if not keywords:
            print(f"\n===== KEYWORD CLUSTERS FOR {chunk_label} =====\n")
            continue

        clusters, term_embeddings = cluster_keywords(keywords, min_clusters=2)
        clusters = filter_by_coherence(clusters, term_embeddings, threshold=0.4)
        print_and_save_clusters(clusters, chunk_label, f)
        all_clusters_by_chunk[chunk_label] = list(clusters.keys())

    parsed, flagged, clean_relations, relation_scores  = query_llm(all_clusters_by_chunk, toc_context,pages,page_embeddings,chunks)
    print("\n===== RUNNING DETERMINISTIC VALIDATOR =====")


    save_concepts(parsed, flagged, f)
    if parsed:
        with open(CONCEPTS_FILE, "w", encoding="utf-8") as jf:
            json.dump(parsed, jf, indent=2)
        print(f"\n✅ Concepts saved to {CONCEPTS_FILE}")

    save_relation_scores(relation_scores, f)
    if clean_relations:
        with open(CLEAN_RELATIONS_FILE, "w", encoding="utf-8") as rf:
            json.dump(clean_relations, rf, indent=2)
            try:
                G_llm = build_networkx_dag(clean_relations)
                plot_dag(G_llm,file_name="dag_llm.png", title="Curriculum DAG — LLM Verified")
                print_dag_summary(G_llm)
            except Exception as e:
                print(f"Error occurred while building DAG: {e}")

        print(f"\n✅ Clean relations saved to {CLEAN_RELATIONS_FILE}")
    
    
    
    print("\n===== DETERMINISTIC VALIDATOR RESULTS =====\n")
    deterministic_results = validate_prerequisite_ordering( parsed,pages,page_embeddings,chunks,DETERMINSITIC_VALIDATOR_OUTPUT)
    print("CURRICULUM ARCHITECTURE EXTRACTION COMPLETE.")


doc.close()
print(f"\nAll text saved to {OUTPUT_FILE}")