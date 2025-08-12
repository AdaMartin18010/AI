#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast and simple duplicate detector for Markdown files.
- Roots: default ["Matter", "Analysis"] or pass dirs via argv
- Method: token set Jaccard similarity
- Threshold: env THRESHOLD (default 0.6)
"""
import os, sys, re, itertools, math
from collections import Counter

WORD_RE = re.compile(r"[\w\-\u4e00-\u9fff]+", re.UNICODE)

def list_md_files(roots):
    files = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith('.md'):
                    files.append(os.path.join(dirpath, fn))
    return files

def tokenize(text):
    return set(w.lower() for w in WORD_RE.findall(text))

def read_tokens(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return tokenize(f.read())
    except Exception:
        return set()

def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def main():
    roots = sys.argv[1:] or ["Matter", "Analysis"]
    thr = float(os.environ.get("THRESHOLD", "0.6"))
    files = list_md_files(roots)
    tokens = {p: read_tokens(p) for p in files}
    pairs = []
    for i, a in enumerate(files):
        ta = tokens[a]
        for b in files[i+1:]:
            s = jaccard(ta, tokens[b])
            if s >= thr:
                pairs.append((s, a, b))
    pairs.sort(reverse=True)
    for s, a, b in pairs[:200]:
        print(f"{s:.3f}\t{a}\t{b}")

if __name__ == "__main__":
    main() 