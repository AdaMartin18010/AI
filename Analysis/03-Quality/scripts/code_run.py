#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, sys
FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)\n```", re.S)

def list_md_files(roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith('.md'):
                    yield os.path.join(dp, fn)

def main():
    roots = sys.argv[1:] or ["Analysis", "Matter"]
    lang_count = {}
    total = 0
    for md in list_md_files(roots):
        with open(md, 'r', encoding='utf-8', errors='ignore') as f:
            t = f.read()
        for lang, _ in FENCE.findall(t):
            total += 1
            key = (lang or 'plain').lower()
            lang_count[key] = lang_count.get(key, 0) + 1
    print(f"code blocks: {total}")
    for k in sorted(lang_count):
        print(f"  {k}: {lang_count[k]}")

if __name__ == '__main__':
    main() 