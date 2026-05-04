"""Aggregate per-cell outputs into Figure 1 + the headline table.

Figure 1: two stacked subplots.
  Top: per-token p95 KL trajectory. x = token offset; y = KL(steer || base).
       Colour by method, linestyle by alpha (solid=1, dashed=2), seed bands
       as thin lines, faceted by model. Horizontal at target_kl=1.
  Bottom: branch-pmass at fork points. x = fork token offset; y = mean pmass
       across held-out prompts; bands = +/- 1 std across seeds.

Table: one row per (model, method), columns = c_star (mean +/- std across seeds),
  KL_p95 @ alpha=1, KL_p95 @ alpha=2, pmass @ alpha=1, pmass @ alpha=2.

Usage:
  python scripts/aggregate.py --runs_root outputs --out figs/
"""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from loguru import logger


@dataclass
class Args:
    runs_root: str = "outputs"
    out: str = "figs"


def load_cells(root: Path) -> list[dict]:
    cells = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        calib = d / "calib.json"
        if not calib.exists():
            continue
        meta = json.loads(calib.read_text())
        traj = json.loads((d / "trajectory.json").read_text())
        pmass = json.loads((d / "pmass.json").read_text())
        cells.append({"id": d.name, **meta, "traj": traj, "pmass": pmass})
    return cells


def make_table(cells: list[dict]) -> pl.DataFrame:
    rows = []
    by_mm = defaultdict(list)
    for c in cells:
        by_mm[(c["model"], c["method"])].append(c)
    for (model, method), group in by_mm.items():
        c_stars = [g["c_star"] for g in group]
        # pmass: mean over fork_points and prompts at each alpha, then across seeds
        for alpha in ("1.0", "2.0"):
            kls = []
            pms = []
            for g in group:
                kls.append(g["traj"]["per_t_p95_kl"][alpha])
                pms.append(g["pmass"]["pmass"][alpha])
            kls_flat = [x for arr in kls for x in arr]
            pms_flat = [x for prompt in pms for arr in prompt for x in arr]
            rows.append({
                "model": model.split("/")[-1],
                "method": method,
                "alpha": float(alpha),
                "c_star_mean": sum(c_stars) / len(c_stars),
                "n_seeds": len(group),
                "kl_p95_mean": sum(kls_flat) / max(len(kls_flat), 1),
                "pmass_mean": sum(pms_flat) / max(len(pms_flat), 1),
            })
    return pl.DataFrame(rows)


def make_figure(cells: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    models = sorted({c["model"] for c in cells})
    methods = sorted({c["method"] for c in cells})
    fig, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 7),
                             sharex="col", squeeze=False)
    cmap = plt.get_cmap("tab10")
    method_color = {m: cmap(i) for i, m in enumerate(methods)}

    for ci, model in enumerate(models):
        ax_kl = axes[0, ci]
        ax_pm = axes[1, ci]
        ax_kl.set_title(model.split("/")[-1])
        ax_kl.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax_kl.set_ylabel("p95 KL(steer || base)")
        ax_pm.set_xlabel("token offset")
        ax_pm.set_ylabel("branch pmass")
        ax_pm.set_ylim(-0.02, 1.02)
        ax_kl.set_yscale("log")

        for method in methods:
            for alpha, ls in [("1.0", "-"), ("2.0", "--")]:
                kls = [c["traj"]["per_t_p95_kl"][alpha]
                       for c in cells if c["model"] == model and c["method"] == method]
                if not kls:
                    continue
                arr = np.array(kls)
                x = np.arange(arr.shape[1])
                ax_kl.plot(x, arr.mean(0), color=method_color[method],
                           linestyle=ls, linewidth=2,
                           label=f"{method} a={alpha}")
                if arr.shape[0] > 1:
                    ax_kl.fill_between(x, arr.min(0), arr.max(0),
                                       color=method_color[method], alpha=0.12)

                pms = [c["pmass"]["pmass"][alpha]
                       for c in cells if c["model"] == model and c["method"] == method]
                if not pms:
                    continue
                # pms: list of (n_seed) of (n_prompt) of (n_fork)
                pms_arr = np.array(pms)  # (n_seed, n_prompt, n_fork)
                fork = cells[0]["pmass"]["fork_points"]
                mean = pms_arr.mean(axis=(0, 1))
                std = pms_arr.std(axis=(0, 1))
                ax_pm.plot(fork, mean, color=method_color[method],
                           linestyle=ls, linewidth=2)
                ax_pm.fill_between(fork, mean - std, mean + std,
                                   color=method_color[method], alpha=0.12)
        if ci == 0:
            ax_kl.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"figure -> {out_path}")


def main(a: Args):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(Path(a.runs_root))
    if not cells:
        raise SystemExit(f"no cells under {a.runs_root}")
    logger.info(f"loaded {len(cells)} cells")

    df = make_table(cells)
    df.write_csv(out / "table.csv")
    md = df.to_pandas().to_markdown(index=False, floatfmt=".3f")
    (out / "table.md").write_text(md)
    logger.info(f"table -> {out/'table.md'}\n{md}")

    make_figure(cells, out / "figure1.png")


if __name__ == "__main__":
    main(tyro.cli(Args))
