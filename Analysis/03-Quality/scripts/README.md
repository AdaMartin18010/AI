# Quality Scripts Usage

- dup_check.py: 检测重复文档对
  - 用法：`python dup_check.py [roots...]`，环境变量 `THRESHOLD`（默认 0.6）
- link_check.py: 检查 Markdown 相对链接有效性
  - 用法：`python link_check.py [roots...]`
- terms_check.py: 检查是否包含“术语表/符号表”
  - 用法：`python terms_check.py [roots...]`
- math_check.py: 检查 `$...$` 使用与 `\(\)/\[\]`
  - 用法：`python math_check.py [roots...]`
- code_run.py: 统计代码块语言与数量（不执行）
  - 用法：`python code_run.py [roots...]`
