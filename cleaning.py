import re

def clean_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\x00-\x09\x0B-\x1F\x7F]', ' ', text)
    text = re.sub(r'[\ufffd\u25a0\u25cf\u2022\u00b7\u2013\u2014\u00a0]', ' ', text)
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'(\. ){2,}', ' ', text)
    text = re.sub(
        r'^(credits|copyright notice|licensing information|creative commons attribution)[^\n]*\n'
        r'(.*?\n)*?(?=\n|\Z)',
        '', text, flags=re.IGNORECASE | re.MULTILINE
    )
    text = re.sub(r'^\s*[\-–]?\s*\d+\s*[\-–]?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\W_]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(grade\s*\d+|unit\s*\d+|lesson\s*\d+|page\s*\d+)\s*$',
                  '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    bad_line_patterns = [
        r'copyright', r'creative commons', r'licen[sc]ed',
        r'open up resources', r'illustrative mathematics',
        r'core knowledge', r'photo credits', r'illustration',
        r'all rights reserved', r'do not reproduce',
        r'may not be reproduced', r'for classroom use', r'©',
    ]

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(re.search(p, lower) for p in bad_line_patterns):
            continue
        alpha_chars = sum(c.isalpha() for c in stripped)
        if alpha_chars >= 5 and alpha_chars / max(len(stripped), 1) > 0.4:
            cleaned_lines.append(stripped)

    return '\n'.join(cleaned_lines)


def clean_for_keybert(text):
    import re
    text = re.sub(r'\[IMAGE \d+:\s*', '', text)
    text = re.sub(r'\]', '', text)
    text = re.sub(r'Text in image:\s*', '', text)
    text = re.sub(r'Visual:\s*', '', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


def is_good_keyword(kw):
    import re
    kw = kw.lower()
    bad_patterns = [
        r'\bstudents?\b', r'\bteacher\b', r'\bactivity\b',
        r'\blesson\b', r'\bwork\b', r'\bdiscussion\b',
        r'\bgroup\b', r'\bpractice\b', r'\bexample\b',
        r'\bproblem\b', r'\bpage\b'
    ]
    if len(kw.split()) > 4:
        return False
    if any(re.search(p, kw) for p in bad_patterns):
        return False
    return True