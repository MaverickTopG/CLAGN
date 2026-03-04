"""
Lint paper text for claim discipline and required citations.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = [
    r"\bdiscover(?:y|ed|ies)?\b.*changing[- ]look",
    r"\bwe discover\b",
    r"\bpipeline detects changing[- ]look",
    r"\bnew changing[- ]look AGN\b",
]

REQUIRED_CITATIONS = [
    'LaMassa2015',  # definition of CLAGN
    'Kelly2009',    # DRW model
    'Wright2010',   # WISE photometric system
]

OPTIONAL_CITATION_GROUPS = [
    ['Sheng2017', 'Graham2020'],
]


def check_paper(tex_path: str = 'paper/clagn_paper.tex') -> dict:
    path = Path(tex_path)
    if not path.exists():
        return {'pass': False, 'errors': [f"Missing {tex_path}"]}

    text = path.read_text(encoding='utf-8')
    errors = []

    # Forbidden claims
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            errors.append(f"Forbidden claim pattern found: {pat}")

    # Required citations
    for cite in REQUIRED_CITATIONS:
        if cite not in text:
            errors.append(f"Missing required citation: {cite}")

    for group in OPTIONAL_CITATION_GROUPS:
        if not any(c in text for c in group):
            errors.append(f"Missing seasonal amplitude citation (one of {group})")

    return {'pass': len(errors) == 0, 'errors': errors}


def main():
    report = check_paper()
    if report['pass']:
        print("CLAIMS CHECK: PASS")
        sys.exit(0)
    print("CLAIMS CHECK: FAIL")
    for err in report['errors']:
        print(f"- {err}")
    sys.exit(1)


if __name__ == '__main__':
    main()
