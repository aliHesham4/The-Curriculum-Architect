import re

def is_toc_page(text):
    toc_signals = [
        r'\btable of contents\b',
        r'\bunit overview\b',
        r'\bsection overview\b',
        r'\bconcept overview\b',
    ]
    has_toc_header = any(re.search(p, text, re.IGNORECASE) for p in toc_signals)

    # TOC lines look like "Section A: Lessons 1–3 .............. 7"
    toc_line_pattern = r'.{5,}[\.\s]{3,}\d+\s*$'
    lines = text.splitlines()
    toc_line_count = sum(1 for line in lines if re.search(toc_line_pattern, line))
    has_toc_structure = toc_line_count >= 3

    return has_toc_header or has_toc_structure
