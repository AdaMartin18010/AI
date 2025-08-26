#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版链接检查脚本 - 集成动态更新机制和自动化质量检查
支持变更检测、依赖分析、增量执行和质量报告
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

class QualityChecker:
    """自动化质量检查器"""
    
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "issues": [],
            "files_checked": 0,
            "broken_links": 0,
            "orphan_anchors": 0,
            "duplicate_anchors": 0,
            "cycles": 0
        }
        
    def detect_changes(self):
        """检测Git变更"""
        try:
            diff = subprocess.check_output(
                ["git", "diff", "--name-status", "origin/main..."], 
                cwd=self.root_dir
            ).decode()
            
            changes = {"added": [], "modified": [], "deleted": []}
            for line in diff.splitlines():
                if line.strip():
                    status, path = line.split("\t", 1)
                    if status == 'A': 
                        changes["added"].append(path)
                    elif status == 'M': 
                        changes["modified"].append(path)
                    elif status == 'D': 
                        changes["deleted"].append(path)
            
            return changes
        except subprocess.CalledProcessError:
            # 如果没有origin/main分支，返回空变更
            return {"added": [], "modified": [], "deleted": []}
    
    def build_crossref_graph(self):
        """构建交叉引用图"""
        refs = defaultdict(set)
        anchors = defaultdict(set)
        
        for md_file in self.root_dir.rglob("*.md"):
            try:
                text = md_file.read_text(encoding='utf-8', errors='ignore')
                
                # 提取锚点
                anchor_pattern = re.compile(r"^#{1,6}\s+(?P<anchor>.+)$", re.MULTILINE)
                for match in anchor_pattern.finditer(text):
                    anchor = match.group('anchor').strip()
                    slug = self.slugify(anchor)
                    anchors[str(md_file)].add(slug)
                
                # 提取链接
                link_pattern = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<link>[^\)]+)\)")
                for match in link_pattern.finditer(text):
                    link = match.group('link')
                    refs[str(md_file)].add(link)
                    
            except Exception as e:
                self.report["issues"].append({
                    "type": "file_read_error",
                    "file": str(md_file),
                    "error": str(e)
                })
        
        return {"refs": dict(refs), "anchors": dict(anchors)}
    
    def slugify(self, title):
        """标准化锚点名称"""
        s = title.strip().lower()
        s = re.sub(r"[^a-z0-9\-\s]", "", s)
        s = re.sub(r"\s+", "-", s)
        return s
    
    def check_file_quality(self, file_path):
        """检查单个文件的质量"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except:
                return [{"type": "encoding_error", "file": str(file_path)}]
        
        # 检查标题层级
        lines = content.split('\n')
        current_level = 0
        anchors = set()
        
        for i, line in enumerate(lines, 1):
            # 检查标题层级连续性
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                if level > current_level + 1:
                    issues.append({
                        "type": "heading_level_skip",
                        "file": str(file_path),
                        "line": i,
                        "message": f"标题层级跳跃: {current_level} -> {level}"
                    })
                current_level = level
                
                # 检查重复锚点
                title = line.lstrip('#').strip()
                slug = self.slugify(title)
                if slug in anchors:
                    issues.append({
                        "type": "duplicate_anchor",
                        "file": str(file_path),
                        "line": i,
                        "message": f"重复锚点: {slug}"
                    })
                anchors.add(slug)
        
        # 检查链接
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        for match in link_pattern.finditer(content):
            text, url = match.groups()
            
            # 跳过外部链接
            if url.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                continue
                
            # 检查内部链接
            if not self.is_valid_internal_link(file_path, url):
                issues.append({
                    "type": "broken_link",
                    "file": str(file_path),
                    "link": url,
                    "text": text
                })
        
        # 检查Mermaid语法
        mermaid_pattern = re.compile(r'```mermaid\s*\n(.*?)\n```', re.DOTALL)
        for match in mermaid_pattern.finditer(content):
            mermaid_code = match.group(1)
            if not self.validate_mermaid(mermaid_code):
                issues.append({
                    "type": "mermaid_error",
                    "file": str(file_path),
                    "message": "Mermaid语法错误"
                })
        
        return issues
    
    def is_valid_internal_link(self, source_file, link):
        """检查内部链接是否有效"""
        source_path = Path(source_file)
        
        # 处理锚点链接
        if '#' in link:
            file_part, anchor_part = link.split('#', 1)
        else:
            file_part, anchor_part = link, None
        
        # 解析文件路径
        if file_part:
            if file_part.startswith('./'):
                target_path = source_path.parent / file_part[2:]
            elif file_part.startswith('../'):
                target_path = source_path.parent / file_part
            else:
                target_path = source_path.parent / file_part
            
            # 检查文件是否存在
            if not target_path.exists():
                return False
            
            # 检查锚点
            if anchor_part:
                try:
                    content = target_path.read_text(encoding='utf-8')
                    anchors = set()
                    for line in content.split('\n'):
                        if line.startswith('#'):
                            title = line.lstrip('#').strip()
                            anchors.add(self.slugify(title))
                    
                    if self.slugify(anchor_part) not in anchors:
                        return False
                except:
                    return False
        else:
            # 页内锚点
            if anchor_part:
                try:
                    content = source_path.read_text(encoding='utf-8')
                    anchors = set()
                    for line in content.split('\n'):
                        if line.startswith('#'):
                            title = line.lstrip('#').strip()
                            anchors.add(self.slugify(title))
                    
                    if self.slugify(anchor_part) not in anchors:
                        return False
                except:
                    return False
        
        return True
    
    def validate_mermaid(self, mermaid_code):
        """验证Mermaid语法（基本检查）"""
        # 基本语法检查
        if not mermaid_code.strip():
            return False
        
        # 检查基本关键字
        keywords = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram']
        has_keyword = any(keyword in mermaid_code for keyword in keywords)
        
        if not has_keyword:
            return False
        
        # 检查括号匹配
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in mermaid_code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                if brackets[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def run_parallel_checks(self, files):
        """并行运行质量检查"""
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.check_file_quality, f) for f in files]
            results = []
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as e:
                    results.append({
                        "type": "check_error",
                        "error": str(e)
                    })
            return results
    
    def generate_report(self, output_file=None):
        """生成质量报告"""
        # 统计问题
        issue_types = defaultdict(int)
        for issue in self.report["issues"]:
            issue_types[issue["type"]] += 1
        
        self.report["summary"] = dict(issue_types)
        
        # 生成人类可读报告
        report_text = f"""# 质量检查报告

**检查时间**: {self.report["timestamp"]}
**检查文件数**: {self.report["files_checked"]}

## 问题统计

"""
        
        for issue_type, count in issue_types.items():
            report_text += f"- **{issue_type}**: {count}个\n"
        
        if self.report["issues"]:
            report_text += "\n## 详细问题\n\n"
            for issue in self.report["issues"][:20]:  # 只显示前20个问题
                report_text += f"- **{issue['type']}**: {issue.get('file', 'N/A')}"
                if 'line' in issue:
                    report_text += f" (第{issue['line']}行)"
                if 'message' in issue:
                    report_text += f" - {issue['message']}"
                report_text += "\n"
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.report, f, indent=2, ensure_ascii=False)
            
            # 保存人类可读报告
            readable_report = output_file.replace('.json', '.md')
            with open(readable_report, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return self.report

def main():
    parser = argparse.ArgumentParser(description='优化版链接检查和质量检查脚本')
    parser.add_argument('--root', default='.', help='根目录路径')
    parser.add_argument('--report', default='quality-report.json', help='报告输出文件')
    parser.add_argument('--changes-only', action='store_true', help='只检查变更的文件')
    parser.add_argument('--fail-fast', action='store_true', help='遇到错误立即失败')
    
    args = parser.parse_args()
    
    checker = QualityChecker(args.root)
    
    # 检测变更
    if args.changes_only:
        changes = checker.detect_changes()
        files_to_check = []
        for file_list in changes.values():
            files_to_check.extend([Path(args.root) / f for f in file_list if f.endswith('.md')])
    else:
        files_to_check = list(Path(args.root).rglob("*.md"))
    
    checker.report["files_checked"] = len(files_to_check)
    
    # 运行检查
    issues = checker.run_parallel_checks(files_to_check)
    checker.report["issues"] = issues
    
    # 生成报告
    report = checker.generate_report(args.report)
    
    # 检查是否应该失败
    if args.fail_fast and report["issues"]:
        critical_issues = [i for i in issues if i["type"] in ["broken_link", "mermaid_error"]]
        if critical_issues:
            print(f"发现{len(critical_issues)}个严重问题，检查失败")
            sys.exit(1)
    
    print(f"质量检查完成，检查了{len(files_to_check)}个文件，发现{len(issues)}个问题")
    print(f"报告已保存到: {args.report}")

if __name__ == "__main__":
    main() 