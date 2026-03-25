import shutil
from pathlib import Path
from typing import Any


class PaperPackageGenerator:
    def __init__(self, config: Any, paper_root: Any = None):
        self.config = config
        self.paper_root = (
            Path(paper_root) if paper_root is not None else Path("paper_package")
        )

    def _latest_run(self) -> Path:
        runs = sorted(Path("runs").glob("*"))
        return runs[-1] if runs else Path("runs")

    def generate_package(self, results_dir: Any = None, output_dir: Any = None) -> str:
        root = Path(output_dir) if output_dir is not None else self.paper_root
        root.mkdir(parents=True, exist_ok=True)

        for name in ["data_cards", "configs", "metrics", "figs"]:
            (root / name).mkdir(parents=True, exist_ok=True)

        src = Path(results_dir) if results_dir is not None else self._latest_run()
        if src.exists():
            for candidate in [
                "results.json",
                "workflow_results.json",
                "metrics.jsonl",
                "config_merged.yaml",
            ]:
                p = src / candidate
                if p.exists():
                    target = (
                        root
                        / ("configs" if candidate.endswith(".yaml") else "metrics")
                        / p.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)

            viz_src = src / "visualizations"
            if viz_src.exists():
                shutil.copytree(viz_src, root / "figs" / src.name, dirs_exist_ok=True)

        return str(root)

    def generate_complete_package(
        self, trainer=None, validation_results=None, checkpoints=None, seed_results=None
    ) -> Path:
        root = self.generate_package()
        return root
