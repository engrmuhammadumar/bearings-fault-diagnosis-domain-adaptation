"""Find physics-branch equations and loss weights in a study folder.

Read-only: does not load checkpoints, execute notebooks, or change study files.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

EXTENSIONS = {".py", ".ipynb", ".json", ".yaml", ".yml", ".toml",
              ".txt", ".md", ".cfg", ".ini", ".log"}
EXCLUDED = {".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv",
            "env", "site-packages", "node_modules", "dataset", "datasets",
            "raw", "raw_data", "c1", "c2", "c3", "c4", "c5", "c6"}
SEARCHES = {
 "loss weights": re.compile(r"(?i)(lambda|weight|alpha|beta|gamma).{0,15}(wear|flute|rul|nll|phys|smooth|cons|mono|ode|reg)|(wear|flute|rul|nll|phys|smooth|cons|mono|ode|reg).{0,15}(lambda|weight)"),
 "complete loss": re.compile(r"(?i)(total.?loss|loss.?total|criterion|objective|losses\s*=|loss_weights)"),
 "wear coefficient": re.compile(r"(?i)(k_eff|wear.?coefficient|archard|kappa|raw.?k|effective.?coefficient)"),
 "positive activation": re.compile(r"(?i)(softplus|relu|exp\s*\(|positive|clamp|min.?rate)"),
 "force proxy": re.compile(r"(?i)(f_eff|force.?(proxy|rms|magnitude|resultant)|dynamometer)"),
 "sliding velocity": re.compile(r"(?i)(sliding.?(speed|velocity)|cutting.?speed|spindle.?speed|10400|10_400|diameter|rpm)"),
 "latent mapping": re.compile(r"(?i)(wear.?(coordinate|channel|state)|latent.*wear|physics.*latent|e_w|z_w|dz.*dt|vector.?field)"),
 "smoothness": re.compile(r"(?i)(smooth|second.?(difference|derivative)|finite.?difference|curvature)"),
 "monotonicity": re.compile(r"(?i)(monotonic|non.?negative.?wear|positive.?rate|cumsum|cummax)"),
 "integration": re.compile(r"(?i)(rk4|runge|odeint|ode.?solver|integrat|delta.?t)"),
}
ASSIGNMENT = re.compile(
 r"(?i)\b([A-Za-z_][A-Za-z0-9_]*(?:lambda|weight|alpha|beta|gamma|wear|flute|rul|nll|phys|smooth|cons|mono|ode|k_eff|kappa)[A-Za-z0-9_]*)\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

def notebook_lines(path):
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    out = []
    for cell_no, cell in enumerate(doc.get("cells", []), 1):
        for line_no, line in enumerate("".join(cell.get("source", [])).splitlines(), 1):
            out.append((f"cell {cell_no}", line_no, line))
        for output in cell.get("outputs", []):
            value = output.get("text", [])
            value = [value] if isinstance(value, str) else value
            for line_no, line in enumerate("".join(value).splitlines(), 1):
                out.append((f"output {cell_no}", line_no, line))
    return out

def text_lines(path):
    try:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return []
    return [("line", i, line) for i, line in enumerate(content.splitlines(), 1)]

def eligible(path):
    parts = {part.lower() for part in path.parts}
    return (path.is_file() and path.suffix.lower() in EXTENSIONS
            and not parts.intersection(EXCLUDED)
            and path.stat().st_size <= 50 * 1024 * 1024)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("pindp_physics_config_audit.md"))
    parser.add_argument("--csv", type=Path, default=Path("pindp_physics_config_matches.csv"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    matches, assignments, scanned = [], [], 0
    for path in sorted(p for p in root.rglob("*") if eligible(p)):
        scanned += 1
        rows = notebook_lines(path) if path.suffix.lower() == ".ipynb" else text_lines(path)
        for location, number, line in rows:
            value = line.strip()
            if not value:
                continue
            for category, pattern in SEARCHES.items():
                if pattern.search(value):
                    matches.append({"category": category, "file": str(path.relative_to(root)),
                                    "location": location, "line": number, "text": value[:500]})
            for found in ASSIGNMENT.finditer(value):
                assignments.append({"name": found.group(1), "value": found.group(2),
                                    "file": str(path.relative_to(root)),
                                    "location": location, "line": number})

    with args.csv.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["category", "file", "location", "line", "text"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(matches)

    report = ["# PINDP-Net physics/configuration audit", "",
              f"- Search root: {root}", f"- Files scanned: {scanned}",
              f"- Relevant matches: {len(matches)}",
              f"- Candidate assignments: {len(assignments)}", "",
              "## Candidate numerical values", "",
              "| Name | Value | File | Location |", "| --- | ---: | --- | --- |"]
    seen = set()
    for item in assignments:
        key = tuple(item.values())
        if key in seen:
            continue
        seen.add(key)
        report.append(f"| {item['name']} | {item['value']} | {item['file']} | {item['location']} {item['line']} |")
    if not assignments:
        report.append("| No candidate found | - | - | - |")

    for category in SEARCHES:
        report += ["", "## " + category.title(), ""]
        selected = [item for item in matches if item["category"] == category]
        if not selected:
            report.append("No match found."); continue
        for item in selected[:100]:
            safe = item["text"].replace("|", "/")
            report.append(f"- {item['file']}, {item['location']} {item['line']}: {safe}")

    report += ["", "## Manual verification", "",
               "1. Prefer the final saved run configuration over code defaults.",
               "2. Confirm that each loss term is actually added to total_loss.",
               "3. A defined but unused lambda is not an active regularizer.",
               "4. Check whether values differ between outer folds.",
               "5. Ignore comments, examples, and abandoned notebook cells.", ""]
    args.output.write_text("\n".join(report), encoding="utf-8")
    print(f"Scanned {scanned} files")
    print(f"Report: {args.output.resolve()}")
    print(f"CSV: {args.csv.resolve()}")

if __name__ == "__main__":
    main()
