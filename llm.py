import re
import json
from sentence_transformers import SentenceTransformer, util
from DAG import build_networkx_dag, plot_dag, print_dag_summary
from config import groq_client
from cleaning import clean_text
from config import doc
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
verification_model = SentenceTransformer("all-MiniLM-L6-v2")



#------------------------------------------------------------------

def build_prompt(all_clusters_by_chunk, toc_context):
    clusters_section = ""
    for chunk_label, cluster_names in all_clusters_by_chunk.items():
        clusters_section += f"\n  {chunk_label}\n"
        for name in cluster_names:
            clusters_section += f"    - {name}\n"

    return f"""
You are a curriculum analyst.
Below is the ONLY source of truth, use the table of contents if present and the topic clusters extracted from each 
section of a curriculum document. Your job is to use your relational reasoning to identify and link educational concepts
and their prerequisite relationships. Pass through all TOC context and cluster names to inform your analysis.
Ignore any cluster that refers to materials, objects, or activities rather than curriculum concepts.
{toc_context}

ALL TOPIC CLUSTERS BY SECTION:
{clusters_section}

Your task:
1. Identify all distinct educational concepts.
2. For each concept, list its prerequisites — concepts a student must 
   understand BEFORE learning it.
3. IMPORTANT: Prerequisites must only come from concepts that also appear 
   in the clusters above. Do not invent external prerequisites.
4. Avoid making concept names too broad or too narrow and DO NOT OUTPUT SIMILAR DUPLICATED CONCEPTS.
5. If a concept has no prerequisites within this curriculum, set prerequisites to [].
6. Avoid including concepts that are relevant to "Assessments", "Materials", "Activities" or "Solving Problems", or other non-conceptual clusters.
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

#-------------------------------------------------------
def build_document_index():
    pages = []
    for page_num in range(len(doc)):
        raw_text = doc[page_num].get_text()
        text     = clean_text(raw_text).strip()
        # Keep all pages; use empty string for very short pages
        pages.append({
            "page": page_num + 1,
            "text": text if len(text) > 20 else ""
        })

    texts      = [p["text"] for p in pages]
    embeddings = verification_model.encode(texts, convert_to_tensor=True)

    return pages, embeddings

# ── Private helper (shared by both queries inside the function) ───────────────

def _get_best_page(query,pages,page_embeddings_np,chunk_embeddings,chunks,top_k_chunks,top_k_pages,page_threshold):
    """
    Two-stage RAG retrieval: chunks → pages.
    Returns the single highest-scoring page above threshold, or None.
    """
    query_emb     = verification_model.encode(query, convert_to_numpy=True)
    chunk_scores  = cosine_similarity([query_emb], chunk_embeddings)[0]
    top_chunk_idx = chunk_scores.argsort()[-top_k_chunks:][::-1]

    best = None
    for chunk_i in top_chunk_idx:
        chunk_pages           = chunks[chunk_i]
        chunk_page_embeddings = [page_embeddings_np[p] for p in chunk_pages]
        page_scores           = cosine_similarity([query_emb], chunk_page_embeddings)[0]
        top_pages_idx         = page_scores.argsort()[-top_k_pages:][::-1]

        for idx in top_pages_idx:
            score = page_scores[idx]
            if score < page_threshold:
                continue
            page_index = chunk_pages[idx]
            text       = pages[page_index]["text"]
            if not text.strip():
                continue
            if best is None or score > best["score"]:
                best = {
                    "page_num":   pages[page_index]["page"],
                    "page_index": page_index,
                    "score":      round(float(score), 3),
                }
    return best

#-------------------------------------------------------
# Verification and saving of LLM output
#-------------------------------------------------------


def verify_concept_in_document(concept_name, pages, embeddings, threshold=0.5):
    concept_emb = verification_model.encode(concept_name, convert_to_tensor=True)
    scores      = util.cos_sim(concept_emb, embeddings)[0]
    best_score  = scores.max().item()
    best_page   = pages[scores.argmax().item()]["page"]

    return {
        "concept":   concept_name,
        "found":     best_score >= threshold,
        "score":     round(best_score, 3),
        "best_page": best_page
    }

def rag_verify_llm_output(parsed, pages, embeddings, threshold=0.47):
    clean   = []
    flagged = []

    for concept in parsed["concepts"]:
        name_result = verify_concept_in_document(concept["name"], pages, embeddings, threshold)

        if not name_result["found"]:
            print(f"  ⚠ Hallucinated concept:    '{concept['name']}' (score: {name_result['score']}, best page: {name_result['best_page']})")
            flagged.append(concept["name"])
            continue

        verified_prereqs = []
        for prereq in concept["prerequisites"]:
            prereq_result = verify_concept_in_document(prereq, pages, embeddings, threshold)
            if prereq_result["found"]:
                verified_prereqs.append(prereq)
            else:
                print(f"  ⚠ Hallucinated prerequisite: '{prereq}' (score: {prereq_result['score']}, best page: {prereq_result['best_page']})")
                flagged.append(prereq)

        clean.append({**concept, "prerequisites": verified_prereqs})

    print(f"\n  ✅ Verified concepts:       {len(clean)}")
    print(f"  ⚠ Flagged hallucinations:  {len(flagged)}")

    return {"concepts": clean}, flagged


#-------------------------------------------------------
#Relation verification layer (LLM + RAG)
#-------------------------------------------------------
def query_llm_relation_verifier(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

def build_relation_verification_prompt( parsed, pages, page_embeddings,chunks,top_k_chunks=2,top_k_pages=2,max_chars=400,
page_threshold=0.45):

    # 🔹 Ensure embeddings are numpy
    if not isinstance(page_embeddings, np.ndarray):
        page_embeddings_np = np.array([emb for emb in page_embeddings])
    else:
        page_embeddings_np = page_embeddings

    # 🔹 Build chunk embeddings (mean of pages)
    chunk_embeddings = []
    for chunk in chunks:
        emb = np.mean([page_embeddings_np[p] for p in chunk], axis=0)
        chunk_embeddings.append(emb)

    # 🔹 Proper structured prompt (JSON enforced)
    prompt = """You are verifying prerequisite relationships between concepts.

For EACH pair, return a JSON object with:
- concept
- prerequisite
- Return a score range between -1 and 1 up to 2 decimal places, where:

+1 = strong evidence prerequisite
0  = unrelated
-1 = strong reverse

Use BOTH:
- the provided evidence
- your reasoning about curriculum structure

Return ONLY a JSON array. No explanation.
{
  "concepts": [
    {
      name: "concept name",
      prerequisites: [
        "prerequisite 1", score: "0.85",
        "prerequisite 2", score: "-0.4"
      ]
    }
  ]
}

For each concept to appear once and below it all its prerequisites with their scores. Now evaluate the concepts and their prerequisites.
"""

    relations = []

    for concept in parsed["concepts"]:
        c_name = concept["name"]

        for prereq in concept["prerequisites"]:

            # 🔹 Encode query
            query = f"What concepts must be understood before {c_name}? Consider: {prereq}"
            best = _get_best_page(
                query              = query,
                pages              = pages,
                page_embeddings_np = page_embeddings_np,
                chunk_embeddings   = chunk_embeddings,
                chunks             = chunks,
                top_k_chunks       = top_k_chunks,
                top_k_pages        = top_k_pages,
                page_threshold     = page_threshold,
            )

            if best is None:
                evidence_text = "No strong evidence found."
            else:
                text          = pages[best["page_index"]]["text"]
                evidence_text = f"(Page {best['page_num']}, score={best['score']:.2f}) {text[:max_chars]}"

            # 🔹 Store relation block
            relations.append(f"""
                Concept: {c_name}
                Prerequisite: {prereq}
                Evidence:
                {evidence_text}
                """)

    # 🔹 Join with clear separators (VERY IMPORTANT)
    prompt += "\n---\n".join(relations)

    return prompt


def transform_relation_scores_to_concepts(relation_scores_json):
    data = json.loads(relation_scores_json)

    # 🔹 Initialize concept map
    concept_map = {}

    # 🔹 Ensure all concepts exist
    for item in data:
        concept_map[item["concept"]] = set()

    # 🔹 Process relations
    for item in data:
        concept = item["concept"]

        for pr in item["prerequisites"]:
            prereq = pr["prerequisite"]
            score  = pr["score"]

            # Ensure prereq exists as concept node
            if prereq not in concept_map:
                concept_map[prereq] = set()

            # ✅ Case 1: Strong prerequisite
            if score >= 0.5:
                concept_map[concept].add(prereq)

            # ❌ Case 2: weak → ignore
            elif -0.5 < score < 0.5:
                continue

            # 🔄 Case 3: reverse relation
            elif score <= -0.5:
                concept_map[prereq].add(concept)

    # 🔹 Convert to required format
    output = {
        "concepts": []
    }

    for concept, prereqs in concept_map.items():
        output["concepts"].append({
            "name": concept,
            "prerequisites": sorted(list(prereqs))
        })

    return output

#-------------------------------------------------------
# Deterministic positional validator ( purely RAG + arithmetic)
#-------------------------------------------------------

def validate_prerequisite_ordering(
    parsed,
    pages,
    page_embeddings,
    chunks,
    output_file,
    top_k_chunks=2,
    top_k_pages=2,
    page_threshold=0.4
):

    if not isinstance(page_embeddings, np.ndarray):
        page_embeddings_np = np.array([np.array(e) for e in page_embeddings])
    else:
        page_embeddings_np = page_embeddings

    chunk_embeddings = [
        np.mean([page_embeddings_np[p] for p in chunk], axis=0)
        for chunk in chunks
    ]

    results = []
    counts = {"valid": 0, "invalid": 0, "unknown": 0}
    valid_map = {}

    for concept in parsed.get("concepts", []):
        c_name = concept["name"]

        if c_name not in valid_map:
            valid_map[c_name] = []

        concept_hit = _get_best_page(
            query=f"Introduction and basic explanation of: {c_name}",
            pages=pages,
            page_embeddings_np=page_embeddings_np,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            top_k_chunks=top_k_chunks,
            top_k_pages=top_k_pages,
            page_threshold=page_threshold,
        )

        for prereq in concept.get("prerequisites", []):

            prereq_hit = _get_best_page(
                query=f"Introduction and basic explanation of: {prereq}",
                pages=pages,
                page_embeddings_np=page_embeddings_np,
                chunk_embeddings=chunk_embeddings,
                chunks=chunks,
                top_k_chunks=top_k_chunks,
                top_k_pages=top_k_pages,
                page_threshold=page_threshold,
            )

            if concept_hit is None or prereq_hit is None:
                status = "unknown"
                page_diff = None
            else:
                page_diff = prereq_hit["page_num"] - concept_hit["page_num"]
                status = "valid" if page_diff <= 0 else "invalid"

            counts[status] += 1

            if status == "valid":
                valid_map[c_name].append(prereq)

            entry = {
                "concept": c_name,
                "prerequisite": prereq,
                "status": status,
                "concept_page": concept_hit["page_num"] if concept_hit else None,
                "prereq_page": prereq_hit["page_num"] if prereq_hit else None,
                "page_difference": page_diff
            }

            results.append(entry)

            # PRINT AS BEFORE
            symbol = {"valid": "✅", "invalid": "❌", "unknown": "❓"}[status]

            c_pg = concept_hit["page_num"] if concept_hit else "?"
            p_pg = prereq_hit["page_num"] if prereq_hit else "?"

            diff = f"  Δ={page_diff:+d}" if page_diff is not None else ""

            print(
                f"  {symbol} [{status.upper():7}]  "
                f"'{prereq}' (p.{p_pg}) → '{c_name}' (p.{c_pg}){diff}"
            )

    print(
        f"\n  ✅ Valid: {counts['valid']}  "
        f"❌ Invalid: {counts['invalid']}  "
        f"❓ Unknown: {counts['unknown']}"
    )

    final_output = {
        "concepts": [
            {
                "name": concept,
                "prerequisites": prereqs
            }
            for concept, prereqs in valid_map.items()
        ]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        try:
            G_deterministic= build_networkx_dag(final_output)
            plot_dag(G_deterministic, file_name="dag_deterministic.png", title="Curriculum DAG — Deterministic Validation")
            print_dag_summary(G_deterministic)
        except Exception as e:
            print(f"Error occurred while building DAG: {e}")

    print(f"\n✅ Valid prerequisite graph saved to {output_file}")
    return {"summary": counts, "results": results}


def query_llm(all_clusters_by_chunk, toc_context,pages, page_embeddings, chunks):
    print("\n===== SENDING ALL CLUSTERS TO LLaMA =====")
    prompt = build_prompt(all_clusters_by_chunk, toc_context)

    if len(prompt) > 20000:
        print("⚠ Prompt is large — consider reducing top_n or chunk count")
        return None

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        parsed = json.loads(raw)
        # Mouse Trap for testing verification: inject a fake concept that should be flagged as hallucinated
        parsed["concepts"].append({
        "name": "quantum entanglement theory",
        "prerequisites": ["relativistic calculus", "wave function collapse"]
        })
        print("  🧪 Injected test concept: 'quantum entanglement theory'")
        #-------------------------------------------------------------------
        print("\n===== RUNNING VERIFICATION CONSTRAINT =====")
        parsed, flagged      = rag_verify_llm_output(parsed, pages, page_embeddings)
        relation_prompt = build_relation_verification_prompt(parsed, pages,page_embeddings,chunks)
        print("\n===== QUERYING LLM FOR RELATION VERIFICATION =====")
        relation_scores = query_llm_relation_verifier(relation_prompt)
        clean_relations = transform_relation_scores_to_concepts(relation_scores)
        

        return parsed, flagged, clean_relations, relation_scores

    

    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  ⚠ LLM error: {e}")
        return None
    
#-------------------------------------------------------
# Saving results in a human-readable format
#-------------------------------------------------------


def save_concepts(parsed,flagged,file_handle):
    header  = "\n\n══════════════════════════════════════════════════════\n"
    header += "   CONCEPTS AND PREREQUISITES FOUND IN DOCUMENT\n"
    header += "══════════════════════════════════════════════════════\n"
    print(header)
    file_handle.write(header)

    if not parsed:
        file_handle.write("  No concepts extracted.\n")
        return

    for concept in parsed.get("concepts", []):
        name       = concept.get("name", "Unknown")
        prereqs    = concept.get("prerequisites", [])
        prereq_str = ", ".join(prereqs) if prereqs else "None"
        line = f"  Concept: {name}\n  Prerequisites: {prereq_str}\n\n"
        print(line, end="")
        file_handle.write(line)
    
    if flagged:
        file_handle.write("\n══════════════════════════════════════════════════════\n")
        file_handle.write("  FLAGGED HALLUCINATIONS\n")
        file_handle.write("══════════════════════════════════════════════════════\n")
        for item in flagged:
            file_handle.write(f"  ⚠ {item}\n")

def save_relation_scores(relation_scores, file_handle):
    header  = "\n\n══════════════════════════════════════════════════════\n"
    header += "   RELATIONSHIP VERIFICATION SCORES\n"
    header += "═════════════════════════════════════════════════════\n"
    print(header)
    file_handle.write(header)

    if not relation_scores:
        file_handle.write("  No relation scores available.\n")
        return
    file_handle.write(json.dumps(relation_scores, indent=2))
    
    print(relation_scores)