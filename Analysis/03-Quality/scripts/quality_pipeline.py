#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run(cmd: list[str]) -> subprocess.CompletedProcess:
	return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main() -> int:
	parser = argparse.ArgumentParser(description="Quality pipeline: link check, optional auto-fix, and reporting.")
	parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[3]), help="project root directory")
	parser.add_argument("--report-json", type=str, default="quality-report.json", help="path to JSON report output")
	parser.add_argument("--report-md", type=str, default="QUALITY.md", help="path to Markdown summary output")
	parser.add_argument("--apply-fix", action="store_true", help="apply automatic link fixes after check")
	parser.add_argument("--verbose", action="store_true", help="print tool outputs")
	args = parser.parse_args()

	root = Path(args.root).resolve()
	scripts_dir = root / "Analysis" / "03-Quality" / "scripts"
	check_script = scripts_dir / "link_check_optimized.py"
	fix_script = scripts_dir / "fix_links.py"

	if not check_script.exists():
		print(f"[ERROR] link_check_optimized.py not found: {check_script}")
		return 2
	if args.apply_fix and not fix_script.exists():
		print(f"[ERROR] fix_links.py not found: {fix_script}")
		return 2

	report_json = root / args.report_json
	report_md = root / args.report_md

	# 1) Link check
	cmd_check = [sys.executable, str(check_script), "--report", str(report_json)]
	res_check = run(cmd_check)
	if args.verbose:
		print(res_check.stdout)

	# Attempt to parse JSON report if produced
	summary = {
		"timestamp": datetime.utcnow().isoformat() + "Z",
		"root": str(root),
		"check": {
			"returncode": res_check.returncode,
			"output_lines": len(res_check.stdout.splitlines()),
		},
		"fix": None,
	}

	data = None
	if report_json.exists():
		try:
			data = json.loads(report_json.read_text(encoding="utf-8", errors="ignore"))
			summary["report_metrics"] = {
				"broken_links": data.get("broken_links", 0),
				"anchors_missing": data.get("anchors_missing", 0),
				"external_issues": data.get("external_issues", 0),
				"files_scanned": data.get("files_scanned", 0),
			}
		except Exception as e:
			summary["report_metrics_error"] = f"failed to parse JSON report: {e}"

	# 2) Optional auto-fix
	fix_stdout = ""
	if args.apply_fix:
		cmd_fix = [sys.executable, str(fix_script), "--apply", "--plan"]
		res_fix = run(cmd_fix)
		fix_stdout = res_fix.stdout
		if args.verbose:
			print(res_fix.stdout)
		summary["fix"] = {"returncode": res_fix.returncode, "output_lines": len(fix_stdout.splitlines())}

	# 3) Markdown summary
	lines = []
	lines.append("# Quality Report Summary")
	lines.append("")
	lines.append(f"- Timestamp: {summary['timestamp']}")
	lines.append(f"- Project root: `{summary['root']}`")
	lines.append("")
	lines.append("## Link Check")
	lines.append(f"- Return code: {summary['check']['returncode']}")
	lines.append(f"- Output lines: {summary['check']['output_lines']}")
	if data:
		lines.append("")
		lines.append("### Metrics")
		lines.append(f"- Broken links: {summary['report_metrics'].get('broken_links', 'N/A')}")
		lines.append(f"- Missing anchors: {summary['report_metrics'].get('anchors_missing', 'N/A')}")
		lines.append(f"- External issues: {summary['report_metrics'].get('external_issues', 'N/A')}")
		lines.append(f"- Files scanned: {summary['report_metrics'].get('files_scanned', 'N/A')}")

	if args.apply_fix:
		lines.append("")
		lines.append("## Auto Fix Plan/Apply")
		lines.append(f"- Return code: {summary['fix']['returncode'] if summary['fix'] else 'N/A'}")
		lines.append(f"- Output lines: {summary['fix']['output_lines'] if summary['fix'] else 'N/A'}")

	report_md.parent.mkdir(parents=True, exist_ok=True)
	report_md.write_text("\n".join(lines), encoding="utf-8")
	print(f"[OK] Summary written to: {report_md}")
	print(f"[OK] JSON report at: {report_json} (if produced)")
	return 0


if __name__ == "__main__":
	sys.exit(main()) 