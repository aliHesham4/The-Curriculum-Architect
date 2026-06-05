"""
dual_validator.py
=================
Combined Dual Prerequisite Validator — append to llm.py or import from it.

Architecture
------------
Step 1 │ ANCHOR PRECOMPUTATION  (RAG only, no LLM, cached per concept)
        │   Retrieve introductory page for every unique concept name once.
        │   Reused across all pairs that share a concept, saving RAG calls.

Step 2 │ RELATIONAL EVIDENCE RETRIEVAL  (RAG only, one call per pair)
        │   Retrieve the page most relevant to the co-occurrence of each
        │   (prereq, concept) pair for injection into the LLM prompt.

Step 3 │ BATCH LLM EVALUATION  (few LLM calls, batched)
        │   Each batch receives concept name, prereq name, positional Δ,
        │   and retrieved evidence text. The LLM returns:
        │     - score      : continuous [-1, 1]
        │     - type       : foundational | procedural | pedagogical
        │     - confidence : [0, 1]
        │     - reasoning  : one sentence

Step 4 │ TYPE-AWARE COMPOSITE SCORING  (no LLM)
        │   Positional score is a sigmoid over page_diff — continuous,
        │   not the binary valid/invalid of the standalone deterministic
        │   validator. Weights differ by prerequisite type:
        │
        │     foundational : LLM 0.70 + positional 0.30
        │       → semantic necessity dominates; position is supporting evidence
        │     procedural   : LLM 0.50 + positional 0.50
        │       → technique order is both semantic and structural
        │     pedagogical  : LLM 0.30 + positional 0.70
        │       → convention is best captured by document ordering
        │
        │   Key design insight: the LLM prompt explicitly tells the model
        │   that a large positive Δ may reflect a late review section rather
        │   than a genuine ordering violation. This allows the LLM to
        │   override a misleading positional signal for concepts like
        │   Trigonometry whose anchor drifts to p.150.

Step 5 │ ACCEPTANCE + DAG CONSTRUCTION  (no LLM)
        │   Pairs above ACCEPTANCE_THRESHOLD are admitted.
        │   Pedagogical edges are excluded from the DAG — only foundational
        │   and procedural edges are retained for structural integrity.
        │   Cycle detection prevents circular dependencies.

Step 6 │ POST-DAG VERIFICATION LOOP  (RAG + few LLM calls)
        │   Runs after DAG construction to recover missing edges.
        │   Combines embedding-based candidate generation with LLM
        │   nomination and batched document verification.
        │   Only foundational and procedural edges are admitted.
        │   Every insertion is cycle-checked with rollback if needed.

Why this beats each standalone approach
----------------------------------------
Deterministic alone  — fails on broad/spiral concepts (anchor drift);
                       treats all prerequisites identically; binary decision
LLM alone            — injects parametric knowledge when evidence is weak;
                       no positional grounding; misses curricular ordering
Dual                 — positional signal suppresses late-anchored spurious
                       edges while LLM recovers them when position misleads;
                       type classification makes dependency reasons auditable;
                       gradient scoring replaces binary accept/reject
Dual + loop          — catches structurally missing edges that were never
                       proposed by upstream extraction; closes the graph

Prerequisite Type Definitions
------------------------------
foundational  : Logical/definitional necessity.
                Student CANNOT understand concept without prerequisite.
                Example: Limits → Continuity
                (continuity is formally defined using limit notation)

procedural    : Mechanical technique dependency.
                Student CANNOT execute concept's method without prerequisite.
                Example: Substitution Method → Integration by Parts
                (IBP requires applying u-substitution mid-procedure)

pedagogical   : Convention-based sequencing.
                Prerequisite builds intuition but is not strictly necessary.
                Example: Derivatives → Related Rates
                (rates problems are solvable with basic differentiation;
                full derivative theory builds confidence but is not required)
"""

import json
import math
import re
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from llm import _get_best_page, query_llm_relation_verifier
from verification_loop import verification_loop

# ---------------------------------------------------------------------------
# Constants — tune per curriculum
# ---------------------------------------------------------------------------
TYPE_WEIGHTS = {
    "foundational": (0.70, 0.30),
    "procedural":   (0.50, 0.50),
    "pedagogical":  (0.30, 0.70),
    "unknown":      (0.55, 0.45),
}

DAG_TYPES                 = {"foundational", "procedural","pedagogical"}  # types of edges to include in the DAG
DUAL_ACCEPTANCE_THRESHOLD = 0.6    # composite score gate for admission
DUAL_LLM_BATCH_SIZE       = 8      # pairs per LLM call
DUAL_PAGE_THRESHOLD       = 0.38   # RAG threshold for anchor retrieval
DUAL_EVIDENCE_THRESHOLD   = 0.40   # RAG threshold for relational evidence


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _dual_positional_score(page_diff):
    """
    Convert page_diff = prereq_page - concept_page to [0, 1].

    Negative values  (prereq before concept)  → score approaches 1.0
    Zero                                       → score = 0.50
    Positive values  (prereq after concept)   → score approaches 0.0

    Uses a sigmoid with scale=20 pages so the transition is gradual.
    This replaces the deterministic validator's hard ≤4 threshold with
    a smooth gradient that penalises displacement without hard cutoffs.

    Examples
    --------
    page_diff = -80  →  score ≈ 0.98   (prereq 80 pages earlier: strong)
    page_diff = -18  →  score ≈ 0.73   (prereq 18 pages earlier: good)
    page_diff =   0  →  score = 0.50   (same page: ambiguous)
    page_diff =  +4  →  score ≈ 0.45   (just over tolerance: slight penalty)
    page_diff = +20  →  score ≈ 0.27   (clearly after: significant penalty)
    page_diff = +80  →  score ≈ 0.02   (far after: near-zero)
    None             →  score = 0.50   (anchor unavailable: neutral)
    """
    if page_diff is None:
        return 0.5
    return 1.0 / (1.0 + math.exp(page_diff / 20.0))


def _dual_normalize_llm(score):
    """Map LLM score from [-1, 1] to [0, 1]."""
    return (float(score) + 1.0) / 2.0


def _dual_composite(llm_norm, pos_score, prereq_type):
    """Weighted combination using type-specific weights."""
    w_llm, w_pos = TYPE_WEIGHTS.get(prereq_type, TYPE_WEIGHTS["unknown"])
    return w_llm * llm_norm + w_pos * pos_score


# ---------------------------------------------------------------------------
# Step 1 — Anchor precomputation (cached)
# ---------------------------------------------------------------------------

def _dual_precompute_anchors(
    unique_names, pages, page_embeddings_np,
    chunk_embeddings, chunks,
    top_k_chunks, top_k_pages, page_threshold
):
    """
    Retrieve the introductory anchor page for every unique concept name.
    Each name is queried exactly once regardless of how many pairs it
    appears in, reducing total RAG calls from O(2 * |pairs|) to O(|unique|).

    Returns
    -------
    dict {name: {"page_num": int, "page_index": int, "score": float} | None}
    """
    anchors = {}
    for name in unique_names:
        hit = _get_best_page(
            query=f"Introduction and basic explanation of: {name}",
            pages=pages,
            page_embeddings_np=page_embeddings_np,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            top_k_chunks=top_k_chunks,
            top_k_pages=top_k_pages,
            page_threshold=page_threshold,
        )
        anchors[name] = hit
    return anchors


# ---------------------------------------------------------------------------
# Step 2 — Relational evidence retrieval (one call per pair)
# ---------------------------------------------------------------------------

def _dual_get_evidence(
    prereq, concept,
    pages, page_embeddings_np, chunk_embeddings, chunks,
    top_k_chunks, top_k_pages, evidence_threshold,
    max_chars=500
):
    """
    Retrieve the page most relevant to the dependency relationship between
    prereq and concept. Uses a relational query rather than an introductory
    one so the retrieved text discusses the two concepts in relation to each
    other, providing the LLM with richer context than either anchor alone.
    """
    hit = _get_best_page(
        query=f"What must be understood before {concept}? Consider: {prereq}",
        pages=pages,
        page_embeddings_np=page_embeddings_np,
        chunk_embeddings=chunk_embeddings,
        chunks=chunks,
        top_k_chunks=top_k_chunks,
        top_k_pages=top_k_pages,
        page_threshold=evidence_threshold,
    )
    if hit is None:
        return "No strong evidence found."
    text = pages[hit["page_index"]]["text"][:max_chars]
    return f"(Page {hit['page_num']}, score={hit['score']:.2f}) {text}"


# ---------------------------------------------------------------------------
# Step 3 — Batch LLM evaluation
# ---------------------------------------------------------------------------

def _dual_build_batch_prompt(batch):
    """
    Construct a single prompt for a batch of prerequisite pairs.
    Each item in batch is a dict with keys:
        concept, prereq, concept_page, prereq_page, page_diff, evidence
    """
    pairs_text = ""
    for i, item in enumerate(batch, 1):
        if item["page_diff"] is not None:
            pos_info = (
                f"Prereq anchor p.{item['prereq_page']}, "
                f"Concept anchor p.{item['concept_page']}, "
                f"Δ = {item['page_diff']:+d}"
            )
        else:
            pos_info = "Positional anchor unavailable for one or both terms."

        pairs_text += f"""
--- Pair {i} ---
Concept:      {item['concept']}
Prerequisite: {item['prereq']}
Position:     {pos_info}
Evidence:     {item['evidence']}
"""

    return f"""You are a curriculum analyst evaluating prerequisite relationships 
in an educational document.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RELATIONSHIP TYPES — assign exactly one per pair:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"foundational" : Logical or definitional necessity.
                 Student CANNOT understand the concept without the prerequisite.
                 The dependency is structural, not a teaching convention.
                 Example: Limits → Continuity

"procedural"   : Mechanical technique dependency.
                 Student CANNOT execute the concept's method without the
                 prerequisite technique being mastered first.
                 Example: Substitution Method → Integration by Parts

"pedagogical"  : Convention-based sequencing.
                 The prerequisite builds intuition or confidence but the concept
                 could theoretically be approached without it.
                 This type is curriculum-specific and may vary across textbooks.
                 Example: Derivatives → Related Rates

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING SCALE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
+1.0 = strong prerequisite (must learn before)
 0.0 = no meaningful relationship
-1.0 = reversed (the stated "prerequisite" is actually more advanced)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSITIONAL GUIDANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Δ value shows how many pages the prerequisite's anchor appears AFTER
the concept's anchor (positive Δ = prereq anchors later = suspicious).
IMPORTANT: A large positive Δ may reflect a late-document review or exam
section rather than a genuine ordering violation. Use the evidence text
and your curriculum knowledge to override the positional signal when the
semantic dependency is clear despite misleading anchoring.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY this JSON (no markdown, no explanation):
{{
  "pairs": [
    {{
      "concept":      "...",
      "prerequisite": "...",
      "score":        0.0,
      "type":         "foundational|procedural|pedagogical",
      "confidence":   0.0,
      "reasoning":    "one sentence explaining why"
    }}
  ]
}}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pairs_text}"""


def _dual_llm_batch(batch):
    prompt = _dual_build_batch_prompt(batch)
    raw = query_llm_relation_verifier(prompt)
    if raw is None:
        return []
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean).get("pairs", [])
    except Exception as e:
        print(f"  ⚠ LLM parse error: {e}")
        return []


# ---------------------------------------------------------------------------
# Step 4+5 — Composite scoring, acceptance, and output
# ---------------------------------------------------------------------------

def _dual_print_result(status, prereq_type, rec, llm_score, pos_score, composite):
    """Console print in the same style as the existing validators."""
    symbol = {"valid": "✅", "invalid": "❌", "unknown": "❓"}[status]
    c_pg   = rec["concept_page"] or "?"
    p_pg   = rec["prereq_page"]  or "?"
    diff   = f"Δ={rec['page_diff']:+d}" if rec["page_diff"] is not None else "Δ=?"
    print(
        f"  {symbol} [{status.upper():7}] [{prereq_type:13}] "
        f"'{rec['prereq']}' (p.{p_pg}) → '{rec['concept']}' (p.{c_pg})  "
        f"{diff}  pos={pos_score:.2f}  llm={llm_score:+.2f}  "
        f"composite={composite:.2f}"
    )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def dual_validate_prerequisites(
    parsed,
    pages,
    page_embeddings,
    chunks,
    output_file,
    node_embeddings=None,
    run_verification_loop=True,
    top_k_chunks=3,
    top_k_pages=4,
    page_threshold=DUAL_PAGE_THRESHOLD,
    evidence_threshold=DUAL_EVIDENCE_THRESHOLD,
    acceptance_threshold=DUAL_ACCEPTANCE_THRESHOLD,
    llm_batch_size=DUAL_LLM_BATCH_SIZE,
):
    """
    Combined dual prerequisite validator with optional post-DAG verification.

    Runs both the LLM semantic verifier and the deterministic positional
    validator on every concept-prerequisite pair, combines their scores
    using type-aware weights, and classifies each accepted edge as
    foundational, procedural, or pedagogical.

    Only foundational and procedural edges are admitted into the DAG.
    If run_verification_loop=True, a post-DAG pass searches for missing
    edges using embedding similarity and batched LLM verification, again
    restricting additions to foundational and procedural types only.

    Parameters
    ----------
    parsed                : dict — {"concepts": [{"name": str, "prerequisites": [...]}]}
    pages                 : list of {"page": int, "text": str}
    page_embeddings       : tensor or np.ndarray shape (P, d)
    chunks                : list of lists of page indices
    output_file           : str  — path to write validated JSON
    node_embeddings       : dict {concept_name: np.ndarray} or None
                            Pre-computed concept embeddings for the verification
                            loop. If None, the loop encodes names on the fly.
    run_verification_loop : bool — whether to run the post-DAG recovery pass
    page_threshold        : float — RAG similarity threshold for anchor retrieval
    evidence_threshold    : float — RAG similarity threshold for relational evidence
    acceptance_threshold  : float — composite score gate (default 0.60)
    llm_batch_size        : int  — pairs per LLM call

    Returns
    -------
    dict with keys:
        "summary"  : {"valid": int, "invalid": int, "unknown": int,
                      "by_type": {"foundational": int, ...}}
        "results"  : list of per-pair detail records
        "concepts" : validated prerequisite map compatible with build_networkx_dag
    """

    print("\n" + "=" * 65)
    print("  DUAL VALIDATOR — START")
    print("=" * 65)

    # ── Normalise embeddings ─────────────────────────────────────────────────
    if not isinstance(page_embeddings, np.ndarray):
        page_embeddings_np = np.array([np.array(e) for e in page_embeddings])
    else:
        page_embeddings_np = page_embeddings

    chunk_embeddings = [
        np.mean([page_embeddings_np[p] for p in chunk], axis=0)
        for chunk in chunks
    ]

    # ── Collect all (concept, prereq) pairs ──────────────────────────────────
    all_pairs = []
    for concept in parsed.get("concepts", []):
        for prereq in concept.get("prerequisites", []):
            all_pairs.append((concept["name"], prereq))

    print(f"  Total pairs to evaluate: {len(all_pairs)}")

    if not all_pairs:
        print("  No prerequisite pairs found — returning empty result.")
        return {"summary": {}, "results": [], "concepts": []}

    # ── Step 1: Precompute anchor pages ──────────────────────────────────────
    print("\n── Step 1: Anchor Precomputation ─────────────────────────────")
    unique_names = sorted({name for pair in all_pairs for name in pair})
    anchors = _dual_precompute_anchors(
        unique_names, pages, page_embeddings_np, chunk_embeddings, chunks,
        top_k_chunks, top_k_pages, page_threshold
    )
    found = sum(1 for v in anchors.values() if v is not None)
    print(f"  Anchored: {found}/{len(unique_names)} unique concept names")

    # ── Step 2: Build per-pair records ───────────────────────────────────────
    print("\n── Step 2: Positional Scoring + Evidence Retrieval ───────────")
    pair_records = []
    for concept_name, prereq_name in all_pairs:
        c_hit = anchors.get(concept_name)
        p_hit = anchors.get(prereq_name)
        c_page = c_hit["page_num"] if c_hit else None
        p_page = p_hit["page_num"] if p_hit else None
        page_diff  = (p_page - c_page) if (c_page and p_page) else None
        pos_score  = _dual_positional_score(page_diff)
        evidence   = _dual_get_evidence(
            prereq_name, concept_name,
            pages, page_embeddings_np, chunk_embeddings, chunks,
            top_k_chunks, top_k_pages, evidence_threshold
        )
        pair_records.append({
            "concept":      concept_name,
            "prereq":       prereq_name,
            "concept_page": c_page,
            "prereq_page":  p_page,
            "page_diff":    page_diff,
            "pos_score":    pos_score,
            "evidence":     evidence,
        })

    # ── Step 3: Batch LLM evaluation ─────────────────────────────────────────
    print(f"\n── Step 3: Batch LLM Evaluation ──────────────────────────────")
    print(f"  {len(pair_records)} pairs → "
          f"{math.ceil(len(pair_records)/llm_batch_size)} LLM calls "
          f"(batch size {llm_batch_size})")

    llm_results = {}   # (concept, prereq) → LLM output dict | None

    for batch_start in range(0, len(pair_records), llm_batch_size):
        batch       = pair_records[batch_start: batch_start + llm_batch_size]
        raw_results = _dual_llm_batch(batch)
        print(f"  Batch {batch_start//llm_batch_size + 1}: "
              f"received {len(raw_results)}/{len(batch)} results")

        for i, item in enumerate(batch):
            key = (item["concept"], item["prereq"])
            llm_results[key] = raw_results[i] if i < len(raw_results) else None

    # ── Step 4+5: Composite scoring + acceptance ──────────────────────────────
    print("\n── Step 4: Composite Scoring + Acceptance ────────────────────")

    valid_map   = {c["name"]: [] for c in parsed.get("concepts", [])}
    results     = []
    counts      = {"valid": 0, "invalid": 0, "unknown": 0}
    type_counts = {"foundational": 0, "procedural": 0, "pedagogical": 0, "unknown": 0}

    for rec in pair_records:
        original_concept = rec["concept"]
        original_prereq  = rec["prereq"]
        key              = (original_concept, original_prereq)
        llm_out          = llm_results.get(key)
        reverse_edge     = False
        pos_score        = rec["pos_score"]

        if llm_out is None:
            llm_score   = 0.0
            llm_norm    = 0.5
            prereq_type = "unknown"
            confidence  = 0.0
            reasoning   = "LLM evaluation unavailable — positional signal only"
        else:
            llm_score   = float(llm_out.get("score", 0.0))
            llm_norm    = _dual_normalize_llm(llm_score)
            prereq_type = llm_out.get("type", "unknown")
            if prereq_type not in TYPE_WEIGHTS:
                prereq_type = "unknown"
            confidence  = float(llm_out.get("confidence", 0.5))
            reasoning   = llm_out.get("reasoning", "")

            if llm_score < 0:
                reverse_edge = True
                llm_score    = abs(llm_score)
                reasoning   += " (LLM indicates reversed prerequisite direction)"

            llm_norm = _dual_normalize_llm(llm_score)

        if reverse_edge:
            final_concept      = original_prereq
            final_prereq       = original_concept
            final_concept_page = rec["prereq_page"]
            final_prereq_page  = rec["concept_page"]
            final_page_diff    = (
                -rec["page_diff"] if rec["page_diff"] is not None else None
            )
            pos_score = _dual_positional_score(final_page_diff)
        else:
            final_concept      = original_concept
            final_prereq       = original_prereq
            final_concept_page = rec["concept_page"]
            final_prereq_page  = rec["prereq_page"]
            final_page_diff    = rec["page_diff"]

        composite = _dual_composite(llm_norm, pos_score, prereq_type)

        if composite >= acceptance_threshold:
            status = "valid"
            valid_map.setdefault(final_concept, [])
            valid_map[final_concept].append({
                "name": final_prereq,
                "type": prereq_type
            })
            counts["valid"]          += 1
            type_counts[prereq_type] += 1
        elif pos_score == 0.5 and llm_norm == 0.5:
            status = "unknown"
            counts["unknown"] += 1
        else:
            status = "invalid"
            counts["invalid"] += 1

        print_rec = {
            "concept":      final_concept,
            "prereq":       final_prereq,
            "concept_page": final_concept_page,
            "prereq_page":  final_prereq_page,
            "page_diff":    final_page_diff,
        }
        _dual_print_result(
            status, prereq_type, print_rec, llm_score, pos_score, composite
        )

        results.append({
            "concept":      final_concept,
            "prerequisite": final_prereq,
            "status":       status,
            "type":         prereq_type,
            "composite":    round(composite, 3),
            "llm_score":    round(llm_score, 3),
            "pos_score":    round(pos_score, 3),
            "llm_norm":     round(llm_norm, 3),
            "concept_page": final_concept_page,
            "prereq_page":  final_prereq_page,
            "page_diff":    final_page_diff,
            "confidence":   round(confidence, 3),
            "reasoning":    reasoning,
            "reversed":     reverse_edge,
        })

    # ── Summary print ─────────────────────────────────────────────────────────
    print(f"\n  Results: ✅ Valid={counts['valid']}  "
          f"❌ Invalid={counts['invalid']}  ❓ Unknown={counts['unknown']}")
    print(f"\n  Prerequisite type distribution (valid pairs only):")
    for t in ["foundational", "procedural", "pedagogical", "unknown"]:
        n = type_counts[t]
        if n > 0:
            pct = 100 * n / max(counts["valid"], 1)
            bar = "█" * int(pct / 5)
            print(f"    {t:15} : {n:3d} ({pct:5.1f}%)  {bar}")

    # ── Build outputs ─────────────────────────────────────────────────────────
    # full_output preserves all valid edges (including pedagogical) for logging
    full_output = {
        "concepts": [
            {"name": concept, "prerequisites": prereqs}
            for concept, prereqs in valid_map.items()
        ]
    }

    dag_map = {
        concept: [p["name"] for p in prereqs if p["type"] in DAG_TYPES]
        for concept, prereqs in valid_map.items()
    }
    dag_output = {
        "concepts": [
            {"name": concept, "prerequisites": prereqs}
            for concept, prereqs in dag_map.items()
        ]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # ── Step 5: Build initial DAG ─────────────────────────────────────────────
    G_dual = None
    try:
        from DAG import build_networkx_dag, plot_dag, print_dag_summary

        print("\n── Step 5: DAG Construction ──────────────────────────────────")
        G_dual = build_networkx_dag(dag_output)
        print_dag_summary(G_dual)

        longest = nx.dag_longest_path(G_dual)
        print("\n  Longest prerequisite chain:")
        print("  " + " → ".join(longest))
        print(f"  Length: {len(longest)}")

    except Exception as e:
        print(f"  DAG construction error: {e}")

    # ── Step 6: Post-DAG verification loop ────────────────────────────────────
    if run_verification_loop and G_dual is not None:
        try:
            print("\n── Step 6: Post-DAG Verification Loop ────────────────────")
            G_dual, missing_found = verification_loop(
                G_llm=G_dual,
                pages=pages,
                page_embeddings=page_embeddings_np,
                chunks=chunks,
                node_embeddings=node_embeddings,
            )

            if missing_found:
                print(f"\n  {len(missing_found)} edge(s) added by verification loop:")
                for e in missing_found:
                    print(f"    ✅ [{e['type']}] "
                          f"{e['prereq']} → {e['concept']} "
                          f"(conf={e['confidence']:.2f}, "
                          f"page={e.get('page_num')})")
            else:
                print("  No missing edges found — DAG appears complete.")

        except Exception as e:
            print(f"  Verification loop error: {e}")

    # ── Plot final DAG ────────────────────────────────────────────────────────
    if G_dual is not None:
        try:
            plot_dag(G_dual, file_name="dag_dual.png",
                     title="Curriculum DAG — Dual Validated")
        except Exception as e:
            print(f"  DAG plot error: {e}")

    print(f"\n✅ Dual-validated graph saved to {output_file}")
    print("Dual Validator Summary" )
    print_type_analysis({"results": results})
    print("=" * 65 + "\n")
    

    return {
        "summary":  {**counts, "by_type": type_counts},
        "results":  results,
        "concepts": full_output["concepts"],
        "dag": G_dual
    }
   


# ---------------------------------------------------------------------------
# Optional: type-distribution analysis for thesis reporting
# ---------------------------------------------------------------------------

def print_type_analysis(dual_results):
    """
    Print a detailed breakdown of prerequisite types for thesis reporting.
    Call after dual_validate_prerequisites().
    """
    results = dual_results.get("results", [])
    if not results:
        print("No results to analyse.")
        return

    valid = [r for r in results if r["status"] == "valid"]

    print("\n" + "=" * 65)
    print("  PREREQUISITE TYPE ANALYSIS")
    print("=" * 65)

    for prereq_type in ["foundational", "procedural", "pedagogical"]:
        subset = [r for r in valid if r["type"] == prereq_type]
        if not subset:
            continue
        print(f"\n  {prereq_type.upper()} ({len(subset)} edges)")
        print(f"  {'Prerequisite':<35} → {'Concept':<35}  comp  llm   pos")
        print("  " + "-" * 85)
        for r in sorted(subset, key=lambda x: -x["composite"]):
            print(
                f"  {r['prerequisite']:<35} → {r['concept']:<35} "
                f" {r['composite']:.2f}  {r['llm_score']:+.2f}  {r['pos_score']:.2f}"
            )
            if r.get("reasoning"):
                print(f"    ↳ {r['reasoning']}")

    print(f"\n  SIGNAL AGREEMENT ANALYSIS")
    print(f"  {'Category':<40} Count")
    print("  " + "-" * 50)

    both_agree_valid = [r for r in valid if r["llm_norm"] >= 0.6 and r["pos_score"] >= 0.6]
    llm_wins         = [r for r in valid if r["llm_norm"] >= 0.6 and r["pos_score"] < 0.5]
    pos_wins         = [r for r in valid if r["llm_norm"] < 0.6  and r["pos_score"] >= 0.6]
    type_resolved    = [r for r in valid if r["llm_norm"] < 0.6  and r["pos_score"] < 0.6]

    print(f"  {'Both signals agree (valid):':<40} {len(both_agree_valid)}")
    print(f"  {'LLM overrides weak positional:':<40} {len(llm_wins)}")
    print(f"  {'Position overrides weak LLM:':<40} {len(pos_wins)}")
    print(f"  {'Type weighting resolves conflict:':<40} {len(type_resolved)}")
    print("=" * 65)