"""Regression checks for machine-independent project paths."""

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCANNED_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md"}
IGNORED_PARTS = {".git", "__pycache__", ".trae", "outputs"}
WINDOWS_MACHINE_ROOT = re.compile(r"(?<![A-Za-z0-9_])[EFef]:[\\/]")
LEGACY_ROOTS = (
    "/share/fandixiaLab/",
    "/root/mzy/",
)


def test_project_files_do_not_embed_machine_specific_paths():
    violations = []

    for path in PROJECT_ROOT.rglob("*"):
        if path == Path(__file__).resolve():
            continue
        if not path.is_file() or path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        if any(part in IGNORED_PARTS for part in path.parts):
            continue

        content = path.read_text(encoding="utf-8", errors="ignore")
        if any(root in content for root in LEGACY_ROOTS) or WINDOWS_MACHINE_ROOT.search(content):
            violations.append(str(path.relative_to(PROJECT_ROOT)))

    assert not violations, "Machine-specific paths found in:\n" + "\n".join(violations)
