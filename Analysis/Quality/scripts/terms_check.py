#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys

def list_md_files(roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith('.md'):
                    yield os.path.join(dp, fn)

def has_section(text, keyword):
    return (f"### {keyword}" in text) or (f"## {keyword}" in text)

def main():
    roots = sys.argv[1:] or ["Analysis", "Matter"]
    total = ok = 0
    for md in list_md_files(roots):
        with open(md, 'r', encoding='utf-8', errors='ignore') as f:
            t = f.read()
        total += 1
        terms = has_section(t, "术语表")
        symbols = has_section(t, "符号表")
        if terms and symbols:
            ok += 1
        else:
            missing = []
            if not terms: missing.append("术语表")
            if not symbols: missing.append("符号表")
            print(f"MISSING {md}: {', '.join(missing)}")
    rate = ok / total if total else 1.0
    print(f"Coverage: {ok}/{total} = {rate:.2%}")

if __name__ == '__main__':
    main() 