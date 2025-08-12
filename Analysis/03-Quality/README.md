# 质量门禁与指标

## 质量门禁（必过）

- 术语与符号一致性（与 `01-Standards/README.md` 对齐）
- 章节与编号规范（≤4 层；Eq/Thm/Lem 编号）
- 图表与引用完整（无断链）
- 代码示例可运行性或限制说明
- 交叉引用有效性（跨目录引用可达）

## 量化指标

- 重复率：Jaccard ≥ 0.6 计为重复候选；全库月度 ≤ 0.15
- 断链率：0
- 术语一致性命中率：≥ 0.95
- 符号表覆盖率：≥ 0.95
- 代码片段执行通过率：≥ 0.9

## 检查脚本（占位）

- `scripts/dup_check.py`：内容相似度检测
- `scripts/link_check.py`：引用有效性检查
- `scripts/terms_check.py`：术语一致性检查
- `scripts/math_check.py`：公式/编号扫描
- `scripts/code_run.py`：示例代码快速执行

## 检查点模板（复制到 `checkpoints/`）

```markdown
# Checkpoint YYYY-MM-DD HH:mm

## 状态
- 完成：
- 未完成：
- 阻塞：

## 变更
- 新增：
- 删除：
- 移动/重命名：

## 指标
- 文档数：
- 交叉引用数：
- 重复候选数：
- 断链数：

## 恢复指引
- 下一步：
- 所需输入：
```
