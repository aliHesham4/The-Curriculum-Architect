import re

def is_toc_page(text):
    # Condition 1: has a TOC header
    toc_headers = [
        r'\btable of contents\b',
        r'\bunit overview\b',
        r'\bsection overview\b',
        r'\bconcept overview\b',
        r'\bbrief contents\b',
        r'\bcontents\b',
    ]
    has_toc_header = any(re.search(p, text, re.IGNORECASE) for p in toc_headers)

    # Condition 2: classic dot-leader structure "Chapter 1 ........ 7"
    toc_line_pattern = r'.{5,}[\.\s]{3,}\d+\s*$'
    lines = text.splitlines()
    toc_line_count = sum(1 for line in lines if re.search(toc_line_pattern, line))
    has_toc_structure = toc_line_count >= 3

    # Condition 3: many lecture/unit/chapter entries (your exact format)
    lecture_pattern = r'(lecture|unit|lesson|chapter|section|part|module)\s+\d+'
    lecture_count = len(re.findall(lecture_pattern, text, re.IGNORECASE))
    has_many_lectures = lecture_count >= 5

    return has_toc_header or has_toc_structure or has_many_lectures