import re
import json
from config import groq_client


def build_prompt(all_clusters_by_chunk, toc_context):
    clusters_section = ""
    for chunk_label, cluster_names in all_clusters_by_chunk.items():
        clusters_section += f"\n  {chunk_label}\n"
        for name in cluster_names:
            clusters_section += f"    - {name}\n"

    return f"""
You are a curriculum analyst.
Use the table of contents if present and the topic clusters extracted from each 
section of a curriculum document. Your job is to use your relational reasoning to identify and link educational concepts
and their prerequisite relationships. Pass through all TOC context and cluster names to inform your analysis.
 Ignore any cluster that refers to materials, objects, or activities rather than math concepts.
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
        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  ⚠ LLM error: {e}")
        return None


def save_concepts(parsed, file_handle):
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