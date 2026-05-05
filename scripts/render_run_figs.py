from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger


@dataclass
class Args:
    run_dir: str
    threshold: float = 0.95
    out_name: str = "figs_auto"
    model_contains: str = ""


def _ensure_single_run_root(run_dir: Path) -> Path:
    root = run_dir.parent / f"_{run_dir.name}_single"
    root.mkdir(parents=True, exist_ok=True)
    link = root / run_dir.name
    if link.is_symlink() or link.exists():
        try:
            if link.resolve() == run_dir.resolve():
                return root
        except Exception:
            pass
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            raise RuntimeError(f"refusing to replace non-file staging path: {link}")
    os.symlink(run_dir.resolve(), link)
    return root


def _run(cmd: list[str], cwd: Path) -> None:
    logger.info("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main(a: Args) -> None:
    run_dir = Path(a.run_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    single_root = _ensure_single_run_root(run_dir)
    out_dir = run_dir / a.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = run_dir / "calib.json"
    if not calib.exists():
        raise SystemExit(f"missing {calib}")

    model_filter = a.model_contains or run_dir.name.split("_", 1)[0]
    common = [
        sys.executable,
        "scripts/survival.py",
        "--runs-root", str(single_root),
        "--out", str(out_dir / "survival"),
        "--window", str(__import__("json").loads(calib.read_text())["window"]),
        "--metric", "pmass_eval",
        "--thresholds", str(a.threshold),
        "--model-contains", model_filter,
    ]
    _run(common, repo_root)
    _run([
        sys.executable,
        "scripts/spaghetti_kl_alive.py",
        "--runs-root", str(single_root),
        "--out", str(out_dir / "spaghetti"),
        "--window", str(__import__("json").loads(calib.read_text())["window"]),
        "--threshold", str(a.threshold),
        "--model-contains", model_filter,
    ], repo_root)
    _run([
        sys.executable,
        "scripts/aggregate.py",
        "--runs-root", str(single_root),
        "--out", str(out_dir / "aggregate"),
        "--window", str(__import__("json").loads(calib.read_text())["window"]),
        "--spaghetti",
        "--color-by-pmass",
        "--kl-only",
        "--model-contains", model_filter,
    ], repo_root)

    pngs = [
        out_dir / "survival" / "survival_pmass_eval.png",
        out_dir / "spaghetti" / "kl_alive_spaghetti.png",
        out_dir / "aggregate" / "figure1_kl_only.png",
    ]
    for png in pngs:
        logger.info(f"PNG -> {png}")


if __name__ == "__main__":
    main(tyro.cli(Args))