import re
import json
from sentence_transformers import SentenceTransformer, util
from config import groq_client, doc
from cleaning import clean_text
verification_model = SentenceTransformer("all-MiniLM-L6-v2")



#-------------------------------------------------------
# Verification and saving of LLM output
#-------------------------------------------------------
def build_document_index():
    pages = []
    for page_num in range(len(doc)):
        raw_text = doc[page_num].get_text()
        text     = clean_text(raw_text).strip()
        if len(text) > 20:
            pages.append({"page": page_num + 1, "text": text})

    texts      = [p["text"] for p in pages]
    embeddings = verification_model.encode(texts, convert_to_tensor=True)

    return pages, embeddings

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

def rag_verify_llm_output(parsed, pages, embeddings, threshold=0.5):
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
4. Avoid making concept names too broad or too narrow and do not create duplicate concepts. Use your judgment to find the right level of granularity.
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


def query_llm(all_clusters_by_chunk, toc_context):
    print("\n===== SENDING ALL CLUSTERS TO LLaMA =====")
    prompt = build_prompt(all_clusters_by_chunk, toc_context)

    if len(prompt) > 20000:
        print("⚠ Prompt is large — consider reducing top_n or chunk count")
        return None

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
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
        pages, embeddings    = build_document_index()
        parsed, flagged      = rag_verify_llm_output(parsed, pages, embeddings)

        return parsed, flagged

    

    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  ⚠ LLM error: {e}")
        return None


def save_concepts(parsed,flagged,file_handle):
    header  = "\n\n══════════════════════════════════════════════════════\n"
    header += "  FINAL CONCEPTS AND PREREQUISITES\n"
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