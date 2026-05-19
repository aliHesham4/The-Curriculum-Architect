import os
import re
import json
from dotenv import load_dotenv
import google.api_core.retry as api_retry
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from DAG import build_networkx_dag, plot_dag, print_dag_summary
from cleaning import clean_text
from config import doc
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
from config import model
from detection import is_toc_page
verification_model = SentenceTransformer("all-MiniLM-L6-v2")
from groq import Groq
load_dotenv(override=True)
groq_client = Groq(api_key=os.getenv("GROQ_KEY"))



#------------------------------------------------------------------

def build_prompt(all_clusters_by_chunk, toc_context):
    clusters_section = ""
    for chunk_label, cluster_names in all_clusters_by_chunk.items():
        clusters_section += f"\n  {chunk_label}\n"
        for name in cluster_names:
            clusters_section += f"    - {name}\n"

    return f"""You are a curriculum analyst. Your task is to extract a clean, 
non-redundant set of teachable concepts and their prerequisite relationships 
from the sources below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 1 — TOPIC CLUSTERS BY SECTION (PRIMARY)
Use these as the main evidence for concept extraction.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{clusters_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 2 — TABLE OF CONTENTS (SECONDARY)
Use only to understand structure and sequencing.
Add a concept from the TOC only if it is entirely absent from the clusters
but is strongly implied by a major section title.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{toc_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCEPT EXTRACTION:
- Extract every teachable concept a student would learn and be tested on 
  individually. Do not under-extract.
- Map raw cluster labels to their proper educational names.
    BAD:  "diff_techniques_cluster_3"
    GOOD: "Chain Rule"
- Prefer specific named concepts over broad categories.
    KEEP: "Chain Rule"   DROP: "Differentiation Techniques"
- Exclude anything that is an activity, exercise, example, problem set,
  exam, review session, or application task. If a cluster represents one 
  of these, skip it entirely.

DEDUPLICATION:
- Merge concepts only when they are the same object at the same level 
  of abstraction.
    MERGE:     "Derivative of sin" + "Derivative of cos"
               → "Derivatives of Trigonometric Functions"
    DO NOT MERGE: "First Fundamental Theorem of Calculus" +
                  "Second Fundamental Theorem of Calculus"
- If two names differ only in phrasing, keep the more specific one.

PREREQUISITES:
- For each concept, list only concepts a student must understand BEFORE 
  learning it, based on the curriculum sources above.
- Prerequisites reflect cognitive dependency, not syllabus order.
  Test: "Can a student learn X without knowing Y?" 
  If yes → Y is NOT a prerequisite.
- CRITICAL: Every prerequisite listed MUST appear as a concept name 
  elsewhere in your output. Never invent external prerequisites.
- If a concept has no prerequisites within this curriculum, 
  set prerequisites to [].

OUTPUT SIZE:
- Extract ALL valid concepts. For a full-semester university course, 
  expect 25–60 concepts.
- Fewer than 20 → you are under-extracting.
- More than 80 → you are over-extracting or not merging duplicates.

Return ONLY valid JSON, no explanation, no markdown:
{{
  "concepts": [
    {{
      "name": "concept name",
      "prerequisites": ["prerequisite 1", "prerequisite 2"]
    }}
  ]
}}"""
#-------------------------------------------------------
def build_document_index():
    pages = []
    for page_num in range(len(doc)):
        raw_text = doc[page_num].get_text()
        text     = clean_text(raw_text).strip()

        # Skip TOC pages — they cause false early-page matches
        if is_toc_page(raw_text):
            pages.append({
                "page": page_num + 1,
                "text": ""  # ← empty string, treated as blank page
            })
            continue
    
        pages.append({
            "page": page_num + 1,
            "text": text if len(text) > 20 else ""
        })

    texts      = [p["text"] for p in pages]
    embeddings = verification_model.encode(texts, convert_to_tensor=True)

    return pages, embeddings

# ── Private helper (shared by both queries inside the function) ───────────────

def _get_best_page(query,pages,page_embeddings_np,chunk_embeddings,chunks,top_k_chunks,top_k_pages,page_threshold,early_bias=0.002):
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
            page_num   = pages[page_index]["page"]
            biased_score= score - (page_num * early_bias)  # Slight bias towards earlier pages
            if best is None or biased_score > best["score"]:
                best = {
                    "page_num":   pages[page_index]["page"],
                    "page_index": page_index,
                    "score":      round(float(score), 3),
                    "biased_score": biased_score
                }
    return best


#-------------------------------------------------------
# Verification and saving of LLM output
#-------------------------------------------------------


def verify_concept_in_document(concept_name, pages, embeddings, threshold):
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

def rag_verify_llm_output(parsed, pages, embeddings, threshold=0.37):
    clean = []
    flagged = []

    # Track all verified standalone concepts
    verified_concepts = set()

    # First pass: verify main concepts
    for concept in parsed["concepts"]:
        name_result = verify_concept_in_document(
            concept["name"],
            pages,
            embeddings,
            threshold
        )

        if not name_result["found"]:
            print(
                f"  ⚠ Hallucinated concept: "
                f"'{concept['name']}' "
                f"(score: {name_result['score']}, "
                f"best page: {name_result['best_page']})"
            )
            flagged.append(concept["name"])
            continue

        verified_concepts.add(concept["name"])

        verified_prereqs = []

        for prereq in concept["prerequisites"]:
            prereq_result = verify_concept_in_document(
                prereq,
                pages,
                embeddings,
                threshold
            )

            if prereq_result["found"]:
                verified_prereqs.append(prereq)

                # IMPORTANT:
                # add prerequisite itself as a verified concept
                verified_concepts.add(prereq)

            else:
                print(
                    f"  ⚠ Hallucinated prerequisite: "
                    f"'{prereq}' "
                    f"(score: {prereq_result['score']}, "
                    f"best page: {prereq_result['best_page']})"
                )
                flagged.append(prereq)

        clean.append({
            **concept,
            "prerequisites": verified_prereqs
        })

    # ------------------------------------------------------------------
    # Add missing prerequisite-only concepts as standalone root nodes
    # ------------------------------------------------------------------

    existing_names = {c["name"] for c in clean}

    missing_prereqs = verified_concepts - existing_names

    for prereq in sorted(missing_prereqs):
        clean.append({
            "name": prereq,
            "prerequisites": []
        })

    print(f"\n  ✅ Verified concepts:       {len(clean)}")
    print(f"  ⚠ Flagged hallucinations:  {len(flagged)}")

    return {"concepts": clean}, flagged

#-------------------------------------------------------
#Relation verification layer (LLM + RAG)
#-------------------------------------------------------
def query_llm_relation_verifier(prompt):
    # ── Primary: Groq ───────────────────────────────────────────────
    try:
        groq_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        print("  ✅ Relation verifier: Groq responded")
        return groq_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Relation verifier Groq failed: {e}")
        print("  🔄 Falling back to Gemini...")

    # ── Fallback: Gemini ────────────────────────────────────────────────
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0
        ), request_options={"retry": api_retry.Retry(maximum=0), "timeout": 200})
        print("  ✅ Relation verifier: Gemini responded")
        return response.text

    except Exception as gemini_e:
        print(f"  ✖ Relation verifier Gemini also failed: {gemini_e}")
        return None

def build_relation_verification_prompt( parsed, pages, page_embeddings,chunks,top_k_chunks=2,top_k_pages=2,max_chars=500,
page_threshold=0.4):

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

+1 = strong prerequisite
0  = unrelated
-1 = reversed prerequisite (the "prerequisite" is actually a more advanced concept that should come after "concept")

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


import json

def transform_relation_scores_to_concepts(relation_scores_json):

    import json

    # Parse JSON string if needed
    if isinstance(relation_scores_json, str):
        relation_scores_json = json.loads(relation_scores_json)

    # Support:
    # { "concepts": [...] }
    # OR [...]
    if isinstance(relation_scores_json, dict):
        data = relation_scores_json.get("concepts", [])

    elif isinstance(relation_scores_json, list):
        data = relation_scores_json

    else:
        raise ValueError("Unexpected relation score format")

    concept_map = {}

    stats = {
        "strong": 0,
        "weak": 0,
        "reverse": 0,
        "total_edges": 0
    }

    for item in data:

        concept_name = item.get("name") or item.get("concept") 

        if not concept_name:
            continue

        if concept_name not in concept_map:
            concept_map[concept_name] = set()

        prereqs = item.get("prerequisites", [])

        for pr in prereqs:

            # ----------------------------
            # HANDLE MULTIPLE FORMATS
            # ----------------------------

            prereq_name = None
            score = 0

            # Case 1:
            # { "name": "...", "score": ... }
            if isinstance(pr, dict):

                prereq_name = (
                    pr.get("name")
                    or pr.get("concept")
                    or pr.get("prerequisite")
                )

                score = float(pr.get("score", 0))

            # Case 2:
            # ["Limits", 0.95]
            elif isinstance(pr, list) and len(pr) >= 2:

                prereq_name = pr[0]
                score = float(pr[1])

            # Case 3:
            # "Limits"
            elif isinstance(pr, str):

                prereq_name = pr
                score = 1.0

            # Skip malformed
            if not prereq_name:
                continue

            stats["total_edges"] += 1

            if prereq_name not in concept_map:
                concept_map[prereq_name] = set()

            # Strong prerequisite
            if score >= 0.45:
                concept_map[concept_name].add(prereq_name)
                stats["strong"] += 1

            # Weak prerequisite
            elif 0 < score < 0.45:
                stats["weak"] += 1

            # Reverse relation
            elif score <= -0.5:
                concept_map[prereq_name].add(concept_name)
                stats["reverse"] += 1

    output = {
        "concepts": [],
        "stats": stats
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
    top_k_chunks=3,
    top_k_pages=4,
    page_threshold=0.38
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
                status = "valid" if page_diff <= 4 else "invalid"

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
        
            import networkx as nx
            longest_path = nx.dag_longest_path(G_deterministic)
            length = len(longest_path)

            print("Longest prerequisite chain:")
            print(" → ".join(longest_path))
            print("Length:", length)
        except Exception as e:
            print(f"Error occurred while building DAG: {e}")

    print(f"\n✅ Valid prerequisite graph saved to {output_file}")
    return {"summary": counts, "results": results}


def query_llm(all_clusters_by_chunk, toc_context, pages, page_embeddings, chunks):

    print("\n===== SENDING ALL CLUSTERS TO LLAMA =====")

    prompt = build_prompt(all_clusters_by_chunk, toc_context)
    print(prompt)

    # if len(prompt) > 20000:
    #     print("⚠ Prompt is large — consider reducing top_n or chunk count")
    #     return None, [], None, []

    raw = None

    # ── Primary: Groq ───────────────────────────────────────────────
    try:
        groq_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1
        )

        raw = groq_response.choices[0].message.content.strip()
        print("  ✅ Groq responded successfully")

    except Exception as e:
        print(f"  ⚠ Groq failed: {e}")
        print("  🔄 Falling back to Gemini...")

        # ── Fallback: Gemini ───────────────────────────────────────
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    top_p=1
                ),
                request_options={
                    "retry": api_retry.Retry(maximum=0),
                    "timeout": 200
                }
            )

            raw = response.text.strip()
            print("  ✅ Gemini responded successfully")

        except Exception as e:
            print(f"  ⚠ Gemini failed: {e}")
            return None, [], None, []

    # ── raw must be set by here ────────────────────────────────────
    if raw is None:
        print("  ✖ No response from any provider.")
        return None, [], None, []

    try:
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        parsed = json.loads(raw)

        # Mouse Trap for testing verification
        parsed["concepts"].append({
            "name": "quantum entanglement theory",
            "prerequisites": [
                "relativistic calculus",
                "wave function collapse"
            ]
        })

        print("  🧪 Injected test concept: 'quantum entanglement theory'")

        #-------------------------------------------------------------------
        print("\n===== RUNNING VERIFICATION CONSTRAINT =====")
        parsed, flagged      = rag_verify_llm_output(parsed, pages, page_embeddings)
        relation_prompt = build_relation_verification_prompt(parsed, pages,page_embeddings,chunks)
        with open("Debugging/relation_prompt.txt", "w", encoding="utf-8") as f:
           f.write(relation_prompt)
        print("\n===== QUERYING LLM FOR RELATION VERIFICATION =====")
        relation_scores = query_llm_relation_verifier(relation_prompt)
        if relation_scores is None:
            print("  ✖ Relation verification failed — skipping.")
            return parsed, flagged, None, []
    
        
        clean_relations = transform_relation_scores_to_concepts(relation_scores)
        

        return parsed, flagged, relation_scores, clean_relations

    

    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error: {e}")
        return None, [], None, []
    except Exception as e:
        print(f"  ⚠ LLM error: {e}")
        return None, [], None, []

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


def verification_loop(G_llm, pages, page_embeddings, chunks):
    """
    POST-DAG verification loop.
    Takes the completed DAG and checks D for missing dependencies
    between concepts that currently have NO edge between them.
    """
    
    if not isinstance(page_embeddings, np.ndarray):
        page_embeddings_np = np.array([emb for emb in page_embeddings])
    else:
        page_embeddings_np = page_embeddings

    chunk_embeddings = [
        np.mean([page_embeddings_np[p] for p in chunk], axis=0)
        for chunk in chunks
    ]

    nodes = list(G_llm.nodes())
    missing_found = []
    
    print("\n===== VERIFICATION LOOP — SCANNING FOR MISSING DEPENDENCIES =====")

    for i, concept_a in enumerate(nodes):
        for concept_b in nodes[i+1:]:

            # Skip if any edge already exists between this pair
            if G_llm.has_edge(concept_a, concept_b):
                continue
            if G_llm.has_edge(concept_b, concept_a):
                continue

            # ── Query D for co-occurrence page ────────────────────────
            hit = _get_best_page(
                query=f"{concept_a} {concept_b}",
                pages=pages,
                page_embeddings_np=page_embeddings_np,
                chunk_embeddings=chunk_embeddings,
                chunks=chunks,
                top_k_chunks=2,
                top_k_pages=2,
                page_threshold=0.40
            )

            if hit is None:
                continue

            page_text = pages[hit["page_index"]]["text"][:500]

            # ── Ask LLM to read the actual page text ──────────────────
            prompt = f"""You are auditing a curriculum prerequisite graph.

These two concepts currently have NO relationship in the graph:
- Concept A: {concept_a}  
- Concept B: {concept_b}

Relevant page from the curriculum document (Page {hit['page_num']}):
\"\"\"{page_text}\"\"\"

Based ONLY on this text:
Does a prerequisite relationship exist between these two concepts?
If yes, which must be learned first?

Reply ONLY with this JSON, no explanation:
{{"dependency_exists": true or false, "first": "A" or "B" or null, "confidence": 0.0_to_1.0}}"""

            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r'```json|```', '', raw).strip()
                result = json.loads(raw)

                if result.get("dependency_exists") and result.get("confidence", 0) >= 0.9:
                    
                    # Determine direction
                    if result.get("first") == "A":
                        prereq, concept = concept_a, concept_b
                    else:
                        prereq, concept = concept_b, concept_a

                    # ── Cycle check before adding ─────────────────────
                    G_llm.add_edge(prereq, concept)
                    if not nx.is_directed_acyclic_graph(G_llm):
                        G_llm.remove_edge(prereq, concept)
                        print(f"  ⚠ Skipped {prereq} → {concept} (would create cycle)")
                        continue

                    missing_found.append({
                        "prereq": prereq,
                        "concept": concept,
                        "confidence": result["confidence"],
                        "evidence_page": hit["page_num"]
                    })
                    print(f"  ✅ Missing dependency added: "
                          f"{prereq} → {concept} "
                          f"(page {hit['page_num']}, "
                          f"confidence {result['confidence']})")

            except Exception:
                continue

    print(f"\n  Verification loop complete.")
    print(f"  Missing dependencies found and added: {len(missing_found)}")
    
    return G_llm, missing_found