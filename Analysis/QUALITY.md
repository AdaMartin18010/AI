# 质量管线运行指南

## 一键运行

```bash
python Analysis/03-Quality/scripts/quality_pipeline.py --verbose --apply-fix \
  --report-json quality-report.json --report-md QUALITY.md
```

- 输出：
  - `quality-report.json`：原始检查报告（由 `link_check_optimized.py` 生成）
  - `QUALITY.md`：汇总摘要（便于人读）

## 参数说明
- `--root`：项目根目录（默认为脚本向上三级）
- `--apply-fix`：运行完检查后自动调用 `fix_links.py --apply --plan`
- `--report-json`：JSON 报告输出路径
- `--report-md`：Markdown 摘要输出路径
- `--verbose`：打印底层脚本输出

## CI 建议
- PR：开启 `--apply-fix` 为 false，仅生成报告并 fail-fast 于断链
- Nightly：`--apply-fix` 打开，产出修复计划，人工复核后合入

## 故障排查
- 若找不到 `link_check_optimized.py` 或 `fix_links.py`，检查路径：`Analysis/03-Quality/scripts/`
- 若外链波动导致报告不稳定，可在脚本中加入域白名单降级策略（未来增强） 