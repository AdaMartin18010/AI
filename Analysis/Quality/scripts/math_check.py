#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, re
INLINE = re.compile(r"\\\(.*?\\\)", re.S)
BLOCK = re.compile(r"\\\[.*?\\\]", re.S)
DOLLAR = re.compile(r"\$+.*?\$+", re.S)

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
    for md in list_md_files(roots):
        with open(md, 'r', encoding='utf-8', errors='ignore') as f:
            t = f.read()
        ni = len(INLINE.findall(t))
        nb = len(BLOCK.findall(t))
        nd = len(DOLLAR.findall(t))
        if nd > 0:
            print(f"WARN  ${nd} segments in {md} (prefer \\( \\) and \\[ \\])")
        if ni + nb == 0:
            # not every doc needs math; just informational
            pass

if __name__ == '__main__':
    main() 