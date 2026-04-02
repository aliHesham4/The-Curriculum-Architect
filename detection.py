import re

def is_toc_page(text):
    toc_signals = [
        r'\btable of contents\b',
        r'\bunit overview\b',
        r'\bsection overview\b',
    ]
    matches = sum(1 for p in toc_signals if re.search(p, text, re.IGNORECASE))
    return matches >= 1

