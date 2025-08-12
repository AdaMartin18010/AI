#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复外部链接标记脚本
"""

import os
import re

def fix_external_links():
    """修复所有文件中的外部链接标记"""
    
    # 遍历Analysis目录
    for root, dirs, files in os.walk('Analysis'):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查是否需要更新
                    needs_update = False
                    new_content = content
                    
                    # 查找指向Matter目录的链接，但没有（外部链接）标记的
                    pattern = r'\[([^\]]+)\]\(\.\.\/\.\.\/Matter\/[^)]+\)(?!（外部链接）)'
                    matches = re.findall(pattern, content)
                    
                    if matches:
                        # 替换所有匹配的链接
                        new_content = re.sub(pattern, r'[\1](\g<0>)（外部链接）', content)
                        needs_update = True
                    
                    # 如果有更新，写回文件
                    if needs_update:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Updated external links: {file_path}")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    fix_external_links()
    print("External links fixed!") 