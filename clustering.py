import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from config import embedder

def normalize(term):
    return " ".join(term.lower().strip().split())


def cluster_keywords(keywords, min_clusters=2):
    if not keywords:
        return {}
    if len(keywords) < 2:
        term, score = keywords[0]
        return {term: [(term, score)]}, {}

    terms  = [normalize(kw) for kw, score in keywords]
    scores = {normalize(kw): score for kw, score in keywords}

    embeddings      = embedder.encode(terms, convert_to_numpy=True)
    sim_matrix      = cosine_similarity(embeddings)
    distance_matrix = np.clip(1 - sim_matrix, 0, None)
    n_clusters      = max(min_clusters, min(int(len(terms) / 3), len(terms) - 1))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)

    # Save embeddings mapped to each term
    term_embeddings = {term: embeddings[i] for i, term in enumerate(terms)}

    raw_clusters = {}
    for term, label in zip(terms, labels):
        term = normalize(term)
        raw_clusters.setdefault(label, []).append((term, scores[term]))

    named_clusters = {}
    for members in raw_clusters.values():
        members_sorted = sorted(members, key=lambda x: x[1], reverse=True)
        named_clusters[members_sorted[0][0]] = members_sorted

    return named_clusters, term_embeddings  # ← pass embeddings out


def cluster_coherence(members, term_embeddings):
    terms = [normalize(kw) for kw, _ in members]

    for t in terms:
        if t not in term_embeddings:
            print(f"⚠️ Missing embedding for: '{t}'")

    emb = np.array([term_embeddings[t] for t in terms if t in term_embeddings])

    if len(emb) < 2:
        return 0

    return cosine_similarity(emb).mean()


def filter_by_coherence(clusters, term_embeddings, threshold=0.3):
    return {
        name: members for name, members in clusters.items()
        if cluster_coherence(members, term_embeddings) >= threshold
    }

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