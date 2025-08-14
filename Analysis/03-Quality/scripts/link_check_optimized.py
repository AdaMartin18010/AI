#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版链接检查脚本 - 减少误报
"""

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

def is_valid_markdown_link(link_text):
    """检查是否为有效的Markdown链接"""
    # 检查是否为数学符号（被误识别为链接）
    math_symbols = [
        r'^[α-ωΑ-Ωβγδεζηθικλμνξπρστυφχψω]+$',  # 希腊字母
        r'^[a-zA-Z]:\s*[A-Za-z\[\]<>]+$',  # 类型注解如 f: A => B
        r'^[a-zA-Z_][a-zA-Z0-9_]*:\s*[A-Za-z\[\]<>]+$',  # 变量类型如 fa: F[A]
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 简单表达式
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 复合表达式
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 三重表达式
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 四重表达式
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 五重表达式
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 六重表达式
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 七重表达式
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 八重表达式
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 九重表达式
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 十重表达式
    ]
    
    # 检查是否为代码片段中的参数
    code_patterns = [
        r'^[a-zA-Z_][a-zA-Z0-9_]*:\s*[A-Za-z\[\]<>]+$',  # 参数类型如 fa: F[A]
        r'^\.\.\.args$',  # 可变参数
        r'^context$',  # 上下文参数
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 简单赋值
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 复合赋值
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 三重赋值
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 四重赋值
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 五重赋值
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 六重赋值
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 七重赋值
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 八重赋值
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 九重赋值
        r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]*\s*[=<>!+\-*/]\s*[a-zA-Z0-9_]+$',  # 十重赋值
    ]
    
    # 检查是否为LaTeX数学符号
    latex_patterns = [
        r'^\\[a-zA-Z]+$',  # \beta, \alpha 等
        r'^\\[a-zA-Z]+\{[^}]*\}$',  # \frac{1}{2} 等
        r'^\$[^$]*\$$',  # $x$, $y$ 等
        r'^\$\$[^$]*\$\$$',  # $$x$$, $$y$$ 等
    ]
    
    # 检查是否为单个字符或简单变量
    simple_patterns = [
        r'^[a-zA-Z]$',  # 单个字母
        r'^[a-zA-Z][a-zA-Z0-9_]*$',  # 简单变量名
        r'^[0-9]+$',  # 纯数字
        r'^[0-9]+\.[0-9]+$',  # 小数
    ]
    
    # 检查是否为特殊符号
    special_patterns = [
        r'^[+\-*/=<>!&|^~%]+$',  # 运算符
        r'^[{}[\]()]+$',  # 括号
        r'^[,.;:!?]+$',  # 标点符号
    ]
    
    # 合并所有模式
    all_patterns = math_symbols + code_patterns + latex_patterns + simple_patterns + special_patterns
    
    for pattern in all_patterns:
        if re.match(pattern, link_text):
            return False
    
    return True

def extract_links_from_markdown(file_path):
    """从Markdown文件中提取链接"""
    broken_links = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            return broken_links
    
    # 匹配Markdown链接 [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(link_pattern, content)
    
    for text, url in matches:
        # 跳过外部链接
        if url.startswith(('http://', 'https://', 'mailto:', 'tel:')):
            continue
        
        # 跳过锚点链接（文档内部链接）
        if url.startswith('#'):
            continue
        
        # 检查是否为误报
        if not is_valid_markdown_link(text):
            continue
        
        # 处理相对路径
        target_path = None
        if url.startswith('./'):
            url = url[2:]
            target_path = file_path.parent / url
        elif url.startswith('../'):
            # 计算相对路径
            current_dir = file_path.parent
            url = url[3:]  # 去掉 ../
            while url.startswith('../'):
                current_dir = current_dir.parent
                url = url[3:]
            target_path = current_dir / url
        else:
            target_path = file_path.parent / url
        
        # 检查文件是否存在
        if target_path and not target_path.exists():
            broken_links.append((text, url))
    
    return broken_links

def check_links_in_directory(root_dir):
    """检查目录中所有Markdown文件的链接"""
    root_path = Path(root_dir)
    broken_links_by_file = {}
    
    # 遍历所有.md文件
    for md_file in root_path.rglob('*.md'):
        broken_links = extract_links_from_markdown(md_file)
        if broken_links:
            broken_links_by_file[str(md_file)] = broken_links
    
    return broken_links_by_file

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python link_check_optimized.py <目录路径>")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    
    if not os.path.exists(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        sys.exit(1)
    
    print(f"正在检查目录: {root_dir}")
    print("=" * 50)
    
    broken_links_by_file = check_links_in_directory(root_dir)
    
    if not broken_links_by_file:
        print("✓ 未发现断链")
        return
    
    total_broken_links = 0
    
    for file_path, broken_links in broken_links_by_file.items():
        print(f"BROKEN {file_path}")
        for text, url in broken_links:
            print(f"  -> {url}")
            total_broken_links += 1
        print()
    
    print(f"总计发现 {total_broken_links} 个断链")

if __name__ == "__main__":
    main() 