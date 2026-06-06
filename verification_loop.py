"""
verification_loop_v5.py
========================
Simplified Post-DAG Verification Loop

Architecture
------------
Stage 1 │ CANDIDATE GENERATION  — no LLM
        │   Embedding cosine similarity to find semantically close
        │   concept pairs that have no existing edge.

Stage 2 │ LLM NOMINATION  — 1 LLM call
        │   Single prompt over the full graph to catch structural
        │   gaps that embedding similarity misses.

Stage 3 │ MERGE + DEDUPLICATION  — no LLM
        │   Union of Stage 1 and Stage 2, existing edges removed.

Stage 4 │ BATCH VERIFICATION + TYPE FILTER  — 2-3 LLM calls
        │   Each candidate pair is evaluated with RAG evidence.
        │   The LLM assigns a score, direction, and prerequisite type.
        │   Only foundational and procedural edges are admitted.
        │   Cycle check on every insertion.


"""

import json
import os
import re
import math
import numpy as np
import networkx as nx
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from llm import _get_best_page
from config import model

load_dotenv(override=True)
groq_client = Groq(api_key=os.getenv("GROQ_KEY"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_K_SEMANTIC      = 2    # neighbours per node in Stage 1
VERIFY_BATCH_SIZE   = 10    # pairs per LLM call in Stage 4
PAGE_THRESHOLD      = 0.32  # RAG retrieval threshold
ACCEPTANCE_THRESHOLD = 0.65 # minimum confidence to admit an edge
DAG_TYPES           = {"foundational", "procedural"}


import google.generativeai as genai
import google.api_core.retry as api_retry

def _llm_call(prompt: str) -> str:

    # ─────────────────────────────────────────────
    # Primary: Gemini
    # ─────────────────────────────────────────────
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                top_p=1
            ),
            request_options={
                "retry": api_retry.Retry(maximum=0),
                "timeout": 20
            }
        )

        print("  ✅ Gemini responded")
        return response.text.strip()

    except Exception as e:
        print(f"  ⚠ Gemini failed: {e}")
        print("  🔄 Falling back to Groq...")

    # ─────────────────────────────────────────────
    # Fallback: Groq
    # ─────────────────────────────────────────────
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=4096,
        )

        print("  ✅ Groq responded")
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  ❌ Groq failed: {e}")
        raise RuntimeError(
            "Both Gemini and Groq failed."
        )


def _parse_json(raw: str) -> dict:
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


# ---------------------------------------------------------------------------
# Stage 1 — Embedding candidate generation
# ---------------------------------------------------------------------------

def _stage1_embedding_candidates(nodes: list,
                                  node_embeddings: np.ndarray,
                                  top_k: int = TOP_K_SEMANTIC) -> set:
    """
    Find semantically close concept pairs with no existing edge.
    Returns a set of frozensets — direction is resolved later.
    """
    sim = cosine_similarity(node_embeddings)
    candidates = set()

    for i, node_a in enumerate(nodes):
        neighbours = sim[i].argsort()[-(top_k + 1):-1][::-1]
        for j in neighbours:
            if i != j:
                candidates.add(frozenset({node_a, nodes[j]}))

    print(f"  Stage 1 — embedding candidates: {len(candidates)}")
    return candidates


# ---------------------------------------------------------------------------
# Stage 2 — DAG-level LLM nomination
# ---------------------------------------------------------------------------

def _stage2_llm_nomination(nodes: list, edges: list) -> set:
    """
    Single LLM call over the full graph to nominate missing edges.
    Catches structural gaps that embedding similarity misses.
    """
    prompt = f"""You are auditing a curriculum prerequisite graph.

Current nodes:
{json.dumps(nodes, indent=2)}

Current edges (prerequisite → concept):
{json.dumps([f"{u} → {v}" for u, v in edges], indent=2)}

Identify concept pairs with NO current edge where a direct prerequisite
relationship is likely missing. Focus only on foundational dependencies
(a student cannot understand the concept without the prerequisite) or
procedural dependencies (a student cannot execute the concept without
the prerequisite technique). Ignore pedagogical or convenience links.

Reply ONLY with this JSON (no markdown, no explanation):
{{
  "candidates": [
    {{"prereq": "concept_name", "concept": "concept_name"}},
    ...
  ]
}}"""

    try:
        raw = _llm_call(prompt)
        result = _parse_json(raw)
        nominated = set()
        for c in result.get("candidates", []):
            p, con = c.get("prereq"), c.get("concept")
            if p and con:
                nominated.add(frozenset({p, con}))
        print(f"  Stage 2 — LLM-nominated candidates: {len(nominated)}")
        return nominated
    except Exception as e:
        print(f"  Stage 2 failed ({e}) — using Stage 1 candidates only.")
        return set()


# ---------------------------------------------------------------------------
# Stage 3 — Merge and clean
# ---------------------------------------------------------------------------

def _stage3_merge(G, candidates_s1: set, candidates_s2: set,
                  nodes: list) -> list:
    """
    Union Stage 1 + Stage 2, remove pairs that already have an edge
    or contain nodes not in the graph.
    """
    node_set = set(nodes)
    clean = []

    for pair in candidates_s1 | candidates_s2:
        a, b = tuple(pair)
        if a not in node_set or b not in node_set:
            continue
        if G.has_edge(a, b) or G.has_edge(b, a):
            continue
        clean.append([a, b])

    print(f"  Stage 3 — merged unique candidates: {len(clean)}")
    return clean


# ---------------------------------------------------------------------------
# Stage 4 — Batch verification with type filtering
# ---------------------------------------------------------------------------

def _stage4_batch_verify(candidates: list,
                          pages, page_embeddings_np,
                          chunk_embeddings, chunks) -> list:
    """
    For each candidate pair, retrieve RAG evidence then ask the LLM to
    verify the relationship, assign direction, type, and confidence.
    Only foundational and procedural edges above ACCEPTANCE_THRESHOLD
    are returned.
    """
    # Gather RAG evidence for every pair first
    enriched = []
    for a, b in candidates:
        hit = _get_best_page(
            query=f"{a} {b}",
            pages=pages,
            page_embeddings_np=page_embeddings_np,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            top_k_chunks=2,
            top_k_pages=2,
            page_threshold=PAGE_THRESHOLD,
        )
        evidence = "No page evidence found."
        page_num = None
        if hit is not None:
            page_num = pages[hit["page_index"]]["page"]
            evidence = (
                f"(Page {page_num}, "
                f"score={hit['score']:.2f}) "
                f"{pages[hit['page_index']]['text'][:400]}"
            )
        enriched.append({"A": a, "B": b,
                         "evidence": evidence, "page_num": page_num})

    # Batch LLM verification
    verified = []
    for batch_start in range(0, len(enriched), VERIFY_BATCH_SIZE):
        batch = enriched[batch_start: batch_start + VERIFY_BATCH_SIZE]

        pairs_text = ""
        for i, item in enumerate(batch, 1):
            pairs_text += f"""
--- Pair {i} ---
A: {item['A']}
B: {item['B']}
Evidence: {item['evidence']}
"""

        prompt = f"""You are verifying missing prerequisite relationships 
in a curriculum graph.

For each pair, determine:
1. Does a prerequisite relationship exist?
2. Which concept comes first (is the prerequisite)?
3. What type is the dependency?
4. How confident are you?

DEPENDENCY TYPES — assign exactly one:
  "foundational" : Student CANNOT understand concept without prerequisite.
                   Example: Limits → Continuity
  "procedural"   : Student CANNOT execute concept without prerequisite technique.
                   Example: Substitution Method → Integration by Parts

Only mark valid=true for foundational or procedural dependencies.
Reject pedagogical or convenience links (valid=false).

Reply ONLY with this JSON (no markdown):
{{
  "pairs": [
    {{
      "A": "...",
      "B": "...",
      "valid": true_or_false,
      "prereq": "A_or_B_or_null",
      "type": "foundational_or_procedural_or_null",
      "confidence": 0.0,
      "reasoning": "one sentence"
    }}
  ]
}}

{pairs_text}"""

        try:
            raw  = _llm_call(prompt)
            data = _parse_json(raw)

            for i, result in enumerate(data.get("pairs", [])):
                if not result.get("valid"):
                    continue
                conf = float(result.get("confidence", 0))
                if conf < ACCEPTANCE_THRESHOLD:
                    continue
                dep_type = result.get("type")
                if dep_type not in DAG_TYPES:
                    continue
                first = result.get("prereq")
                a = result.get("A") or batch[i]["A"]
                b = result.get("B") or batch[i]["B"]
                if first == "A":
                    prereq, concept = a, b
                elif first == "B":
                    prereq, concept = b, a
                else:
                    continue

                verified.append({
                    "prereq":   prereq,
                    "concept":  concept,
                    "type":     dep_type,
                    "confidence": conf,
                    "page_num": batch[i]["page_num"],
                    "reasoning": result.get("reasoning", ""),
                })
                print(f"  ✔ [{dep_type}] {prereq} → {concept} "
                      f"(conf={conf:.2f}, page={batch[i]['page_num']})")

        except Exception as e:
            print(f"  Stage 4 batch error ({e}) — skipping batch.")
            continue

    print(f"\n  Stage 4 — verified edges (foundational + procedural): "
          f"{len(verified)}")
    return verified


# ---------------------------------------------------------------------------
# DAG insertion with cycle guard
# ---------------------------------------------------------------------------

def _insert_edges(G, verified: list) -> list:
    """
    Insert verified edges in descending confidence order.
    Roll back any edge that introduces a cycle.
    """
    added = []
    verified.sort(key=lambda x: x["confidence"], reverse=True)

    for e in verified:
        prereq, concept = e["prereq"], e["concept"]
        if G.has_edge(prereq, concept) or G.has_edge(concept, prereq):
            continue

        G.add_edge(prereq, concept,
                   type=e["type"], confidence=e["confidence"])

        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(prereq, concept)
            print(f"  ⚠  Rollback: {prereq} → {concept} "
                  f"(would create cycle)")
            continue

        added.append(e)
        print(f"  ✅ Added [{e['type']}] {prereq} → {concept}")

    return added


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def verification_loop(G_llm, pages, page_embeddings, chunks,
                      node_embeddings: dict = None):
    """
    Simplified Post-DAG Verification Loop v5.
    Adds only foundational and procedural edges to the DAG.

    Parameters
    ----------
    G_llm           : networkx.DiGraph — DAG after primary validation
    pages           : list of {"text": str, "page": int}
    page_embeddings : list or np.ndarray of shape (P, d)
    chunks          : list of lists of page indices
    node_embeddings : dict {concept_name: np.ndarray} or None

    Returns
    -------
    G_llm         : updated DAG
    missing_found : list of dicts describing every added edge
    """
    if not isinstance(page_embeddings, np.ndarray):
        page_embeddings_np = np.array(list(page_embeddings))
    else:
        page_embeddings_np = page_embeddings

    chunk_embeddings = [
        np.mean([page_embeddings_np[p] for p in chunk], axis=0)
        for chunk in chunks
    ]

    nodes = list(G_llm.nodes())
    edges = list(G_llm.edges())

    print("\n" + "=" * 65)
    print("  VERIFICATION LOOP v5 — START")
    print(f"  Nodes: {len(nodes)}   Existing edges: {len(edges)}")
    print("=" * 65)

    # Resolve node embeddings
    if node_embeddings is None:
        print("  Encoding concept names on the fly...")
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        emb_matrix = enc.encode(nodes, show_progress_bar=False)
    else:
        emb_matrix = np.array([node_embeddings[n] for n in nodes])

    print("\n── Stage 1: Embedding Candidate Generation ──────────────────")
    candidates_s1 = _stage1_embedding_candidates(nodes, emb_matrix)

    print("\n── Stage 2: DAG-Level LLM Nomination ────────────────────────")
    candidates_s2 = _stage2_llm_nomination(nodes, edges)

    print("\n── Stage 3: Merge & Deduplicate ─────────────────────────────")
    merged = _stage3_merge(G_llm, candidates_s1, candidates_s2, nodes)

    if not merged:
        print("  No candidates after merge — DAG appears complete.")
        return G_llm, []

    print("\n── Stage 4: Batch Verification + Type Filter ────────────────")
    verified = _stage4_batch_verify(
        merged, pages, page_embeddings_np, chunk_embeddings, chunks
    )

    if not verified:
        print("  No candidates confirmed.")
        return G_llm, []

    print("\n── Inserting Verified Edges ──────────────────────────────────")
    missing_found = _insert_edges(G_llm, verified)

    print("\n" + "=" * 65)
    print(f"  VERIFICATION LOOP v5 — COMPLETE")
    print(f"  Edges added : {len(missing_found)}")
    print(f"  Final edges : {G_llm.number_of_edges()}")
    print(f"  DAG valid   : {nx.is_directed_acyclic_graph(G_llm)}")
    print("=" * 65 + "\n")

    return G_llm, missing_found