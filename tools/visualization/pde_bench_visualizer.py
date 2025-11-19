import os
from pathlib import Path
from typing import Optional
import json


class PDEBenchVisualizer:
    """
    Minimal visualizer to generate a comprehensive HTML report for PDEBench.

    This scans an experiment output directory for common visualization images
    and writes a compact HTML summary under a target directory.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _read_metrics(self, exp_dir: Path) -> Optional[dict]:
        try:
            metrics_path = exp_dir / "test_results.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def create_comprehensive_report(self, experiment_root: str) -> Path:
        """
        Create a single HTML page aggregating available visualizations.
        - Looks for images under `<experiment_root>/visualizations`.
        - Reads `test_results.json` if present.
        - Writes `index.html` in `self.output_dir` and returns its path.
        """
        exp_dir = Path(experiment_root)
        viz_dir = exp_dir / "visualizations"
        images = []
        if viz_dir.exists():
            for p in sorted(viz_dir.glob("*.png")):
                images.append(p.name)

        metrics = self._read_metrics(exp_dir) or {}
        title = f"PDEBench Report - {exp_dir.name}"

        html_lines = [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            f"  <meta charset=\"utf-8\"><title>{title}</title>",
            "  <style>",
            "    body{font-family:Arial, sans-serif;margin:24px}",
            "    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}",
            "    .card{border:1px solid #ddd;border-radius:8px;padding:8px}",
            "    img{max-width:100%;height:auto;border-radius:4px}",
            "    h1{margin:0 0 12px 0}",
            "    .metric{font-size:14px;color:#444}",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>{title}</h1>",
            "  <h2>Metrics Summary</h2>",
            "  <div class=\"metric\">",
        ]
        if metrics:
            try:
                final = metrics.get("final_test_metrics", metrics)
                for k, v in final.items():
                    html_lines.append(f"    <div><strong>{k}</strong>: {v}</div>")
            except Exception:
                html_lines.append("    <div>No available metrics</div>")
        else:
            html_lines.append("    <div>Test metrics file not found</div>")
        html_lines += ["  </div>", "  <h2>Visualization Images</h2>", "  <div class=\"grid\">"]

        for img in images:
            html_lines += [
                "    <div class=\"card\">",
                f"      <div>{img}</div>",
                f"      <img src=\"{img}\" alt=\"{img}\">",
                "    </div>",
            ]

        html_lines += ["  </div>", "</body>", "</html>"]

        out_html = self.output_dir / "index.html"
        with open(out_html, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))

        # Optionally copy images
        try:
            if viz_dir.exists():
                for p in viz_dir.glob("*.png"):
                    dst = self.output_dir / p.name
                    if not dst.exists():
                        dst.write_bytes(p.read_bytes())
        except Exception:
            pass

        return out_html