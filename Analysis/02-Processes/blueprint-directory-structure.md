# 目录重构蓝图（三层）

## 约束

- 层级 ≤ 3；同级文件 ≤ 30；命名 `NN-Topic`（英文）

## Analysis 目标骨架（建议）

- `01-01-Standards/`
- `02-02-Processes/`
- `03-03-Quality/`
- `10-10-AI/`
- `20-20-Mathematics/`
- `30-30-FormalMethods/`
- `40-40-ComputerScience/`
- `50-50-ProgrammingLanguage/`
- `60-60-SoftwareEngineering/`
- `70-70-EmergingTechnologies/`
- `80-80-KnowledgeGraph/`
- `90-Theory/`

## 映射策略（Matter → Analysis）

- 原始素材 `Matter/**`：保持不动；在 `Analysis/**` 产出“精炼版”并建立交叉引用
- 同名或近似主题：合并为单一“主文档”，其余作为附录或归档
- 在 `Analysis/03-Quality/mappings.json` 记录旧→新路径

## 执行顺序（建议）

1. 确认主题归属（10-AI/Math/Formal/SE 等）
2. 生成迁移清单（含回滚脚本思路）
3. 分批迁移并修复引用
4. 质量验收（断链=0、重复率下降、门禁通过）
