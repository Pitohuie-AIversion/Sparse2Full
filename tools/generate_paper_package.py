import shutil
from pathlib import Path
from typing import Any

class PaperPackageGenerator:
    def __init__(self, config: Any, paper_root: Path):
        self.config = config
        self.paper_root = paper_root

    def _latest_run(self) -> Path:
        runs = sorted(Path("runs").glob("*"))
        return runs[-1] if runs else Path("runs")

    def generate_package(self) -> Path:
        root = self.paper_root
        root.mkdir(parents=True, exist_ok=True)
        figs = root / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        latest = self._latest_run()
        viz_src = latest / "test_visualizations"
        if viz_src.exists():
            dst = figs / latest.name
            shutil.copytree(viz_src, dst, dirs_exist_ok=True)
        ckpt_dst = root / "checkpoints"
        ckpt_dst.mkdir(exist_ok=True)
        for name in ["best.ckpt", "last.ckpt", "final_model.pth"]:
            p = latest / name
            if p.exists():
                shutil.copy2(p, ckpt_dst / p.name)
        return root

    def generate_complete_package(self, trainer=None, validation_results=None, checkpoints=None, seed_results=None) -> Path:
        root = self.generate_package()
        return root
