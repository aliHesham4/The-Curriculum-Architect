import fitz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Open PDF
doc = fitz.open(r"D:\GUC\The Curriculum Architect\Dataset\Math Curriculum For Children.pdf")
total_pages = len(doc)
chunk_size = 50  # pages per chunk

chunks = []

# Collect chunks of cleaned text
for chunk_start in range(0, total_pages, chunk_size):
    chunk_end = min(chunk_start + chunk_size, total_pages)
    chunk_text = ""

    for page_number in range(chunk_start, chunk_end):
        page = doc[page_number]
        text = page.get_text()
        if len(text.strip()) < 20:  # skip empty pages
            continue
        chunk_text += f"\n\n===== PAGE {page_number + 1} =====\n{text}"

    # Clean text: remove extra spaces and line breaks
    chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
    chunks.append(chunk_text)

doc.close()

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',  # remove common words
    ngram_range=(1, 4),    # 1-4 words as phrases
    max_df=0.85,           # ignore terms that appear in >85% of chunks
    min_df=1               # ignore terms that appear in <1 chunk
)

# Fit TF-IDF on chunks
tfidf_matrix = vectorizer.fit_transform(chunks)
feature_names = vectorizer.get_feature_names_out()

top_n = 15  # top keywords per chunk

# Extract top TF-IDF terms per chunk
for i, chunk_vec in enumerate(tfidf_matrix):
    scores = chunk_vec.toarray()[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    top_terms = [feature_names[idx] for idx in top_indices]

    print(f"\n\n===== CHUNK PAGES {i*chunk_size+1} - {min((i+1)*chunk_size, total_pages)} =====")
    for term in top_terms:
        print(term)
