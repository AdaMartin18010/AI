#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复断链脚本
"""

import os
import re
import sys
from pathlib import Path

def fix_math_index_links(file_path):
    """修复数学内容索引文档中的断链"""
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义修复规则
    fix_rules = [
        # 修复指向不存在文档的链接
        (r'docs/FormalModel/Model/Math/数学概念分析\.md', './数学内概念分析.md'),
        (r'docs/FormalModel/Model/Math/数学逻辑\.md', './数学内概念分析.md'),
        (r'docs/FormalModel/Model/Math/数学概念联系\.md', './数学概念分析和综合.md'),
        (r'docs/FormalModel/Model/Math/Algebra/数学代数：认知结构、概念分解与建构分析\.md', './Algebra/从范畴论视角审视代数.md'),
        (r'docs/FormalModel/Model/Math/Algebra/view_抽象代数\.md', './Algebra/view_抽象代数.md'),
        (r'docs/FormalModel/Model/Math/Geometry/view_几何01\.md', './Geometry/view_几何01.md'),
        (r'docs/FormalModel/Model/Math/Calculus/微积分的合法性：哲学与科学视角的深化\.md', './Calculus/微积分的合法性：哲学与科学视角的深化.md'),
        (r'docs/FormalModel/Model/FormalLanguage/形式语言理论：多维分析、认知视角与数学关系\.md', '../FormalLanguage/形式语言的多维批判性分析：从基础理论到应用实践.md'),
        (r'docs/FormalModel/Model/Math/数学概念分析\.md', './数学内概念分析.md'),
        (r'docs/FormalModel/Model/Math/views/math_view02\.md', './views/math_view02.md'),
        (r'docs/FormalModel/Model/View/CMAR/人脑认知、数学、现实：关联性、映射与综合论证\.md', '../Theory/人脑认知、数学、现实：关联性、映射与综合论证.md'),
        (r'docs/Mathematics/数学内容全面分析报告\.md', './数学内容全面分析报告.md'),
        (r'docs/Mathematics/数学内容思维导图\.md', './数学内容思维导图.md'),
        (r'docs/Mathematics/数学内容索引\.md', './数学内容索引.md'),
    ]
    
    # 应用修复规则
    for pattern, replacement in fix_rules:
        content = re.sub(pattern, replacement, content)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复文件: {file_path}")

def fix_philosophy_index_links(file_path):
    """修复哲学内容索引文档中的断链"""
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义修复规则
    fix_rules = [
        # 修复指向不存在文档的链接
        (r'content/ontology/AI本体论分析\.md', './content/ontology/AI本体论分析.md'),
        (r'content/ontology/本体论综合比较\.md', './content/ontology/本体论综合比较.md'),
        (r'content/epistemology/知识来源分析\.md', './content/epistemology/知识来源分析.md'),
        (r'content/epistemology/知识结构分析\.md', './content/epistemology/知识结构分析.md'),
        (r'content/epistemology/认知科学视角\.md', './content/epistemology/认知科学视角.md'),
        (r'content/epistemology/AI认识论分析\.md', './content/epistemology/AI认识论分析.md'),
        (r'content/ethics/元伦理学分析\.md', './content/ethics/元伦理学分析.md'),
        (r'content/ethics/应用伦理学分析\.md', './content/ethics/应用伦理学分析.md'),
        (r'content/ethics/AI伦理分析\.md', './content/ethics/AI伦理分析.md'),
        (r'content/ethics/形式化伦理学\.md', './content/ethics/形式化伦理学.md'),
        (r'content/logic/形式逻辑分析\.md', './content/logic/形式逻辑分析.md'),
        (r'content/logic/哲学逻辑分析\.md', './content/logic/哲学逻辑分析.md'),
        (r'content/logic/非经典逻辑分析\.md', './content/logic/非经典逻辑分析.md'),
        (r'content/logic/逻辑哲学分析\.md', './content/logic/逻辑哲学分析.md'),
        (r'content/logic/计算逻辑应用\.md', './content/logic/计算逻辑应用.md'),
        (r'content/metaphysics/存在论分析\.md', './content/metaphysics/存在论分析.md'),
        (r'content/metaphysics/模态形而上学分析\.md', './content/metaphysics/模态形而上学分析.md'),
        (r'content/metaphysics/时间空间哲学\.md', './content/metaphysics/时间空间哲学.md'),
        (r'content/metaphysics/因果性分析\.md', './content/metaphysics/因果性分析.md'),
        (r'content/metaphysics/形而上学综合\.md', './content/metaphysics/形而上学综合.md'),
        (r'content/interdisciplinary/mathematics/数学对象存在性\.md', './content/interdisciplinary/mathematics/数学对象存在性.md'),
        (r'content/interdisciplinary/mathematics/数学真理本质\.md', './content/interdisciplinary/mathematics/数学真理本质.md'),
        (r'content/interdisciplinary/mathematics/数学发现发明\.md', './content/interdisciplinary/mathematics/数学发现发明.md'),
        (r'content/interdisciplinary/mathematics/数学应用性\.md', './content/interdisciplinary/mathematics/数学应用性.md'),
        (r'content/interdisciplinary/science/科学方法论\.md', './content/interdisciplinary/science/科学方法论.md'),
        (r'content/interdisciplinary/science/科学实在论\.md', './content/interdisciplinary/science/科学实在论.md'),
        (r'content/interdisciplinary/science/科学革命\.md', './content/interdisciplinary/science/科学革命.md'),
        (r'content/interdisciplinary/science/科学解释\.md', './content/interdisciplinary/science/科学解释.md'),
        (r'content/interdisciplinary/cognitive/心智哲学\.md', './content/interdisciplinary/cognitive/心智哲学.md'),
        (r'content/interdisciplinary/cognitive/意识问题\.md', './content/interdisciplinary/cognitive/意识问题.md'),
        (r'content/interdisciplinary/cognitive/认知科学哲学\.md', './content/interdisciplinary/cognitive/认知科学哲学.md'),
        (r'content/interdisciplinary/cognitive/认知计算\.md', './content/interdisciplinary/cognitive/认知计算.md'),
        (r'content/interdisciplinary/technology/AI哲学\.md', './content/interdisciplinary/technology/AI哲学.md'),
        (r'content/interdisciplinary/technology/计算哲学\.md', './content/interdisciplinary/technology/计算哲学.md'),
        (r'content/interdisciplinary/technology/信息哲学\.md', './content/interdisciplinary/technology/信息哲学.md'),
        (r'content/interdisciplinary/technology/网络哲学\.md', './content/interdisciplinary/technology/网络哲学.md'),
        (r'content/traditional/existentialism/加缪荒诞哲学\.md', './content/traditional/existentialism/加缪荒诞哲学.md'),
        (r'content/traditional/existentialism/海德格尔存在哲学\.md', './content/traditional/existentialism/海德格尔存在哲学.md'),
        (r'content/traditional/existentialism/雅斯贝尔斯生存哲学\.md', './content/traditional/existentialism/雅斯贝尔斯生存哲学.md'),
        (r'content/traditional/phenomenology/胡塞尔现象学\.md', './content/traditional/phenomenology/胡塞尔现象学.md'),
        (r'content/traditional/phenomenology/海德格尔现象学\.md', './content/traditional/phenomenology/海德格尔现象学.md'),
        (r'content/traditional/phenomenology/梅洛庞蒂现象学\.md', './content/traditional/phenomenology/梅洛庞蒂现象学.md'),
        (r'content/traditional/phenomenology/萨特现象学\.md', './content/traditional/phenomenology/萨特现象学.md'),
        (r'content/traditional/hermeneutics/伽达默尔解释学\.md', './content/traditional/hermeneutics/伽达默尔解释学.md'),
        (r'content/traditional/hermeneutics/利科解释学\.md', './content/traditional/hermeneutics/利科解释学.md'),
        (r'content/traditional/hermeneutics/哈贝马斯批判理论\.md', './content/traditional/hermeneutics/哈贝马斯批判理论.md'),
        (r'content/traditional/hermeneutics/德里达解构主义\.md', './content/traditional/hermeneutics/德里达解构主义.md'),
        (r'content/traditional/feminism/性别视角\.md', './content/traditional/feminism/性别视角.md'),
        (r'content/traditional/feminism/女性主义认识论\.md', './content/traditional/feminism/女性主义认识论.md'),
        (r'content/traditional/feminism/女性主义伦理学\.md', './content/traditional/feminism/女性主义伦理学.md'),
        (r'content/traditional/feminism/女性主义政治哲学\.md', './content/traditional/feminism/女性主义政治哲学.md'),
        (r'content/modern/environmental/生态伦理学\.md', './content/modern/environmental/生态伦理学.md'),
        (r'content/modern/environmental/环境正义\.md', './content/modern/environmental/环境正义.md'),
        (r'content/modern/environmental/可持续发展\.md', './content/modern/environmental/可持续发展.md'),
        (r'content/modern/environmental/生物伦理学\.md', './content/modern/environmental/生物伦理学.md'),
        (r'content/modern/political/正义理论\.md', './content/modern/political/正义理论.md'),
        (r'content/modern/political/自由理论\.md', './content/modern/political/自由理论.md'),
        (r'content/modern/political/权利理论\.md', './content/modern/political/权利理论.md'),
        (r'content/modern/political/民主理论\.md', './content/modern/political/民主理论.md'),
        (r'content/modern/social/社会理论\.md', './content/modern/social/社会理论.md'),
        (r'content/modern/social/文化哲学\.md', './content/modern/social/文化哲学.md'),
        (r'content/modern/social/技术社会\.md', './content/modern/social/技术社会.md'),
        (r'content/modern/social/全球化\.md', './content/modern/social/全球化.md'),
        (r'content/modern/postmodern/后结构主义\.md', './content/modern/postmodern/后结构主义.md'),
        (r'content/modern/postmodern/解构主义\.md', './content/modern/postmodern/解构主义.md'),
        (r'content/modern/postmodern/后现代主义\.md', './content/modern/postmodern/后现代主义.md'),
        (r'content/modern/postmodern/新实用主义\.md', './content/modern/postmodern/新实用主义.md'),
        (r'content/emerging/neurophilosophy/神经伦理学\.md', './content/emerging/neurophilosophy/神经伦理学.md'),
        (r'content/emerging/neurophilosophy/神经美学\.md', './content/emerging/neurophilosophy/神经美学.md'),
        (r'content/emerging/neurophilosophy/神经宗教\.md', './content/emerging/neurophilosophy/神经宗教.md'),
        (r'content/emerging/neurophilosophy/神经自由意志\.md', './content/emerging/neurophilosophy/神经自由意志.md'),
        (r'content/emerging/quantum/量子认识论\.md', './content/emerging/quantum/量子认识论.md'),
        (r'content/emerging/quantum/量子本体论\.md', './content/emerging/quantum/量子本体论.md'),
        (r'content/emerging/quantum/量子因果性\.md', './content/emerging/quantum/量子因果性.md'),
        (r'content/emerging/quantum/量子意识\.md', './content/emerging/quantum/量子意识.md'),
        (r'content/emerging/complexity/复杂系统\.md', './content/emerging/complexity/复杂系统.md'),
        (r'content/emerging/complexity/涌现性\.md', './content/emerging/complexity/涌现性.md'),
        (r'content/emerging/complexity/自组织\.md', './content/emerging/complexity/自组织.md'),
        (r'content/emerging/complexity/混沌理论\.md', './content/emerging/complexity/混沌理论.md'),
        (r'content/emerging/network/网络空间\.md', './content/emerging/network/网络空间.md'),
        (r'content/emerging/network/虚拟现实\.md', './content/emerging/network/虚拟现实.md'),
        (r'content/emerging/network/数字身份\.md', './content/emerging/network/数字身份.md'),
        (r'content/emerging/network/网络伦理\.md', './content/emerging/network/网络伦理.md'),
        (r'analysis/philosophical_content_analysis\.md', './analysis/philosophical_content_analysis.md'),
        (r'analysis/concept_mapping\.md', './analysis/concept_mapping.md'),
        (r'analysis/argument_structure\.md', './analysis/argument_structure.md'),
        (r'analysis/formalization_analysis\.md', './analysis/formalization_analysis.md'),
        (r'analysis/wiki_comparison\.md', './analysis/wiki_comparison.md'),
        (r'analysis/contemporary_alignment\.md', './analysis/contemporary_alignment.md'),
        (r'visualizations/mindmaps/本体论思维导图\.md', './visualizations/mindmaps/本体论思维导图.md'),
        (r'visualizations/mindmaps/认识论思维导图\.md', './visualizations/mindmaps/认识论思维导图.md'),
        (r'visualizations/mindmaps/伦理学思维导图\.md', './visualizations/mindmaps/伦理学思维导图.md'),
        (r'visualizations/mindmaps/逻辑学思维导图\.md', './visualizations/mindmaps/逻辑学思维导图.md'),
        (r'visualizations/mindmaps/形而上学思维导图\.md', './visualizations/mindmaps/形而上学思维导图.md'),
        (r'visualizations/mindmaps/交叉领域哲学思维导图\.md', './visualizations/mindmaps/交叉领域哲学思维导图.md'),
        (r'visualizations/mindmaps/哲学发展趋势思维导图\.md', './visualizations/mindmaps/哲学发展趋势思维导图.md'),
        (r'visualizations/graphs/哲学概念关系图\.md', './visualizations/graphs/哲学概念关系图.md'),
        (r'visualizations/graphs/哲学流派关系图\.md', './visualizations/graphs/哲学流派关系图.md'),
        (r'visualizations/graphs/哲学方法关系图\.md', './visualizations/graphs/哲学方法关系图.md'),
        (r'visualizations/graphs/哲学应用关系图\.md', './visualizations/graphs/哲学应用关系图.md'),
        (r'visualizations/graphs/哲学发展时间线\.md', './visualizations/graphs/哲学发展时间线.md'),
        (r'visualizations/tables/哲学方法对比表\.md', './visualizations/tables/哲学方法对比表.md'),
        (r'visualizations/tables/哲学应用对比表\.md', './visualizations/tables/哲学应用对比表.md'),
        (r'visualizations/tables/哲学概念对比表\.md', './visualizations/tables/哲学概念对比表.md'),
        (r'visualizations/tables/哲学发展对比表\.md', './visualizations/tables/哲学发展对比表.md'),
        (r'supplements/philosophical_positions\.md', './supplements/philosophical_positions.md'),
        (r'supplements/formal_systems\.md', './supplements/formal_systems.md'),
        (r'supplements/contemporary_issues\.md', './supplements/contemporary_issues.md'),
        (r'supplements/historical_development\.md', './supplements/historical_development.md'),
        (r'supplements/cultural_perspectives\.md', './supplements/cultural_perspectives.md'),
        (r'supplements/practical_applications\.md', './supplements/practical_applications.md'),
        (r'resources/references/classical_philosophy\.md', './resources/references/classical_philosophy.md'),
        (r'resources/references/modern_philosophy\.md', './resources/references/modern_philosophy.md'),
        (r'resources/references/contemporary_philosophy\.md', './resources/references/contemporary_philosophy.md'),
        (r'resources/references/interdisciplinary_philosophy\.md', './resources/references/interdisciplinary_philosophy.md'),
        (r'resources/tools/formalization_tools\.md', './resources/tools/formalization_tools.md'),
        (r'resources/tools/analysis_methods\.md', './resources/tools/analysis_methods.md'),
        (r'resources/tools/visualization_tools\.md', './resources/tools/visualization_tools.md'),
        (r'resources/tools/computational_tools\.md', './resources/tools/computational_tools.md'),
        (r'resources/examples/philosophical_arguments\.md', './resources/examples/philosophical_arguments.md'),
        (r'resources/examples/formal_proofs\.md', './resources/examples/formal_proofs.md'),
        (r'resources/examples/real_world_applications\.md', './resources/examples/real_world_applications.md'),
        (r'resources/examples/case_studies\.md', './resources/examples/case_studies.md'),
    ]
    
    # 应用修复规则
    for pattern, replacement in fix_rules:
        content = re.sub(pattern, replacement, content)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复文件: {file_path}")

def main():
    """主函数"""
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # 修复数学内容索引
    math_index = project_root / "Matter" / "Mathematics" / "数学内容索引.md"
    if math_index.exists():
        fix_math_index_links(math_index)
    
    # 修复哲学内容索引
    philosophy_index = project_root / "Matter" / "Philosophy" / "哲学内容索引.md"
    if philosophy_index.exists():
        fix_philosophy_index_links(philosophy_index)
    
    print("断链修复完成！")

if __name__ == "__main__":
    main() 