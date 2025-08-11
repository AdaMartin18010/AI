#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, sys
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

def list_md_files(roots):
    files = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith('.md'):
                    files.append(os.path.join(dp, fn))
    return files

def is_external(href):
    return href.startswith('http://') or href.startswith('https://')

def check_file(md):
    broken = []
    with open(md, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    for href in LINK_RE.findall(txt):
        if '#' in href:
            base = href.split('#', 1)[0]
        else:
            base = href
        if not base or is_external(base):
            continue
        cand = os.path.normpath(os.path.join(os.path.dirname(md), base))
        if not os.path.exists(cand):
            broken.append(href)
    return broken

def main():
    roots = sys.argv[1:] or ["Analysis", "Matter"]
    all_broken = 0
    for md in list_md_files(roots):
        b = check_file(md)
        if b:
            all_broken += len(b)
            print(f"BROKEN {md}")
            for href in b:
                print(f"  -> {href}")
    if all_broken == 0:
        print("All links OK")

if __name__ == '__main__':
    main() 