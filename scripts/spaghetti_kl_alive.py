"""KL-vs-survival spaghetti: per-prompt KL trajectory colored by alive/dead.

Each rollout is a thin line. KL(t) is plotted; the line is colored by whether
the rollout is currently 'alive' (pmass at the nearest fork s<=t is >= threshold)
or 'dead' (it has dropped below threshold at some s<=t -- death is irreversible).

Right-censoring: if the rollout EOS'd before t (gen_len < t), the line stops at
gen_len (drawn as a small terminal dot) -- it is 'complete' not 'dead'.

Panels: one per alpha. Title: KL ceiling (calibration target) + n trajectories.

Reading guide: at alpha=1, KL is bounded near the calibration target by
construction (~1 nat). Lines staying blue/green => model still coherent under
that KL budget. Red lines => model collapsed even though KL stayed in budget
(steering hit a degenerate region without raising KL much).

Usage:
  python scripts/spaghetti_kl_alive.py \
    --runs-root outputs_qwen35_w512_v3 \
    --out figs_qwen35_w512_kl_alive \
    --window 512 \
    --alphas 0.0 0.25 0.5 0.75 1.0 1.5 2.0 4.0 \
    --threshold 0.8 \
    --model-contains Qwen3.5-0.8B
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

try:
    import seaborn as sns
    sns.set_theme(context="notebook", style="whitegrid", palette="deep", font_scale=0.9)
    plt.rcParams.update({
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
except Exception:
    plt.style.use("ggplot")


@dataclass
class Args:
    runs_root: str = "outputs_qwen35_w512_v3"
    out: str = "figs_qwen35_w512_kl_alive"
    window: int = 512
    alphas: tuple[str, ...] = ("0.0", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "4.0")
    threshold: float = 0.8       # pmass < threshold = dead
    metric: str = "pmass_eval"   # 'pmass_eval' is paired with KL prompts
    model_contains: str = "Qwen3.5-0.8B"
    kl_log: bool = True
    roll: int = 11               # smooth KL a bit so the spaghetti is readable


def load_cell(d: Path, alpha: str, T: int):
    """Load one cell: returns list of (kl_traj, pmass_per_fork, fork_points, gen_len)."""
    calib = d / "calib.json"
    traj_p = d / "trajectory.json"
    pm_p = d / "pmass.json"
    if not (calib.exists() and traj_p.exists() and pm_p.exists()):
        return []
    traj = json.loads(traj_p.read_text())
    pm = json.loads(pm_p.read_text())
    if not pm.get("computed", True):
        return []
    fork = pm["fork_points"]
    kl_per_prompt = traj.get("per_prompt_per_t_kl", {}).get(alpha, [])
    pmass_per_prompt = pm.get("pmass_eval", {}).get(alpha, [])
    glens = pm.get("gen_lens_eval", {}).get(alpha, [])
    if not kl_per_prompt or not pmass_per_prompt:
        return []
    out = []
    for i, (kl, pmv) in enumerate(zip(kl_per_prompt, pmass_per_prompt)):
        gl = int(glens[i]) if i < len(glens) else len(kl)
        out.append((np.asarray(kl, dtype=float), np.asarray(pmv, dtype=float), fork, gl))
    return out


def alive_mask_for_t(pmass_per_fork: np.ndarray, fork: list[int],
                     T: int, threshold: float, gen_len: int) -> np.ndarray:
    """Return per-token (T,) mask: True=alive, False=dead. Uses nearest-fork-<=t.
    Once dead at some fork, dead forever after that fork. Right-censoring at gen_len."""
    # walking running-min over forks, then broadcast to per-token via "nearest fork <= t"
    rmin = np.minimum.accumulate(np.where(np.isnan(pmass_per_fork), np.inf, pmass_per_fork))
    dead_at_fork = rmin < threshold
    # for each token t, find largest fork s such that fork[s] <= t
    fork_arr = np.array(fork)
    alive = np.ones(T, dtype=bool)
    for t in range(min(T, gen_len)):
        # idx of last fork <= t
        idx = int(np.searchsorted(fork_arr, t, side="right") - 1)
        if idx >= 0 and dead_at_fork[idx]:
            alive[t] = False
    # tokens beyond gen_len: censored, mark via separate signal (we'll just truncate display)
    return alive


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def main(a: Args):
    root = Path(a.runs_root)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cells = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    cells = [d for d in cells if a.model_contains in d.name]
    logger.info(f"found {len(cells)} cells in {root} matching {a.model_contains!r}")

    n_panels = len(a.alphas)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.4),
                             sharey=True, squeeze=False)
    color_alive = to_rgba("#1a9850", 0.65)   # green, translucent so overlap stays readable
    color_dead = to_rgba("#d7191c", 0.55)    # stronger red so all-dead panels are visible

    summary_rows = []
    for j, alpha in enumerate(a.alphas):
        ax = axes[0, j]
        all_trajs = []
        for d in cells:
            all_trajs.extend(load_cell(d, alpha, a.window))
        n = len(all_trajs)
        n_died = 0
        n_censored = 0
        median_rows = []
        dead_segments = []
        alive_segments = []
        dead_colors = []
        alive_colors = []
        # build all dead segments first, then alive segments on top
        for kl, pmv, fork, gl in all_trajs:
            T = min(len(kl), gl)
            kl = _rolling_mean(kl[:T], a.roll)
            median_rows.append(np.pad(kl, (0, max(0, a.window - T)), constant_values=np.nan)[: a.window])
            alive = alive_mask_for_t(pmv, fork, T, a.threshold, gl)
            xs = np.arange(T)
            if T < 2:
                continue
            pts = np.stack([xs, kl], axis=1).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_alive = alive[:-1]
            dead_segments.extend([seg for seg, al in zip(segs, seg_alive) if not al])
            alive_segments.extend([seg for seg, al in zip(segs, seg_alive) if al])
            dead_colors.extend([color_dead] * int((~seg_alive).sum()))
            alive_colors.extend([color_alive] * int(seg_alive.sum()))
            if not alive.all():
                n_died += 1
            if gl < a.window:
                n_censored += 1
                ax.scatter([gl - 1], [kl[-1]], s=6, c="black", marker="|",
                           alpha=0.6, zorder=3)

        if dead_segments:
            ax.add_collection(LineCollection(dead_segments, colors=dead_colors, linewidths=0.9, zorder=1))
        if alive_segments:
            ax.add_collection(LineCollection(alive_segments, colors=alive_colors, linewidths=0.9, zorder=2))
        if median_rows:
            med = np.nanmedian(np.asarray(median_rows), axis=0)
            med_x = np.arange(len(med))
            if np.nanmax(np.abs(med)) < 1e-6:
                med = med + 1e-4
            ax.plot(med_x, med, color="black", lw=1.3, alpha=0.9, zorder=4)

        ax.axhline(1.0, color="black", lw=0.7, ls=":", label="KL=1 calib target")
        ax.set_title(rf"$\alpha={alpha}$  n={n}  died={n_died}  cens={n_censored}")
        ax.set_xlabel("token t")
        if j == 0:
            ax.set_ylabel("per-token KL")
        if a.kl_log:
            ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xlim(-5, a.window + 5)
        ax.autoscale_view()
        # data-driven y-lim sanity
        # rely on auto
        summary_rows.append({"alpha": alpha, "n": n, "n_died": n_died, "n_censored": n_censored})

    # legend on first panel only
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=color_alive, lw=2, label=f"alive (pmass >= {a.threshold})"),
        Line2D([0],[0], color=color_dead, lw=2, label=f"dead (pmass < {a.threshold})"),
        Line2D([0],[0], color="black", marker="|", lw=0, label="EOS (right-censored)"),
        Line2D([0],[0], color="black", lw=0.7, ls=":", label="KL=1 calib target"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=7, frameon=True)

    fig.suptitle(
        f"KL trajectories coloured by survival (pmass < {a.threshold} = dead). "
        f"Model: {a.model_contains}. window={a.window}.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_p = out / "kl_alive_spaghetti.png"
    fig.savefig(out_p, dpi=160, bbox_inches="tight")
    logger.info(f"figure -> {out_p}")

    from tabulate import tabulate
    md = tabulate(summary_rows, headers="keys", tablefmt="pipe")
    (out / "kl_alive_summary.md").write_text(md)
    logger.info("\n" + md)


if __name__ == "__main__":
    main(tyro.cli(Args))
