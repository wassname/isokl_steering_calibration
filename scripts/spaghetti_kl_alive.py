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
    --runs-root outputs/qwen35_w512_v3 \
    --out figs/qwen35_w512_kl_alive \
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
    sns.set_theme(context="notebook", style="white", palette="deep", font_scale=0.9)
    plt.rcParams.update({
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "#faf7f2",
        "axes.facecolor": "#faf7f2",
        "savefig.facecolor": "#faf7f2",
    })
except Exception:
    plt.style.use("default")


@dataclass
class Args:
    runs_root: str = "outputs/qwen35_w512_v3"
    out: str = "figs/qwen35_w512_kl_alive"
    window: int = 512
    alphas: tuple[str, ...] = ("0.0", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "4.0")
    threshold: float = 0.8       # pmass < threshold = dead
    metric: str = "pmass_eval"   # 'pmass_eval' is paired with KL prompts
    model_contains: str = "Qwen3.5-0.8B"
    kl_log: bool = False
    roll: int = 65               # smooth KL a lot so the spaghetti is readable
    line_lw: float = 0.5         # per-trajectory linewidth -- thick enough to see
    line_alpha: float | None = None  # if None, scale by cohort size: clamp(8/n, .2, .7)
    normalize_kl: bool = True    # divide each KL trajectory by p95(KL[:calib_tokens])
    calib_tokens: int = 20       # window over which to compute the per-traj p95 denom


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


def _panel_xmax(max_gen_len: int, window: int) -> float:
    if max_gen_len <= 0:
        return float(window)
    return float(min(window, max(20, int(max_gen_len * 1.05))))


def _kl_ymax(values: list[float]) -> float:
    if not values:
        return 1.1
    finite = np.asarray([v for v in values if np.isfinite(v) and v >= 0.0], dtype=float)
    if finite.size == 0:
        return 1.1
    return float(max(1e-3, np.nanpercentile(finite, 99) * 1.4))


def main(a: Args):
    root = Path(a.runs_root)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cells = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    cells = [d for d in cells if a.model_contains in d.name]
    logger.info(f"found {len(cells)} cells in {root} matching {a.model_contains!r}")

    n_panels = len(a.alphas)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.4),
                             sharey=False, squeeze=False)
    # alpha-per-line is computed per panel from the cohort size so n=32 is not a paintbrush
    # paper-friendly Wong-ish pair: muted teal-blue (alive) vs warm orange (dead).
    # colourblind-safe, prints well, doesn't scream "stop sign".
    GREEN = "#3a7ca5"   # steel teal -- "alive / in budget"
    RED = "#d97706"     # warm amber -- "dead / out of budget"

    summary_rows = []
    panel_state: list[dict] = []  # collected per-panel info; ylim applied after loop
    all_visible_kls: list[float] = []
    for j, alpha in enumerate(a.alphas):
        ax = axes[0, j]
        all_trajs = []
        for d in cells:
            all_trajs.extend(load_cell(d, alpha, a.window))
        n = len(all_trajs)
        # compute per-line alpha for this panel (more lines -> more transparent)
        if a.line_alpha is not None:
            la = float(a.line_alpha)
        else:
            la = float(np.clip(8.0 / max(n, 1), 0.20, 0.70))
        color_alive = to_rgba(GREEN, la)
        color_dead = to_rgba(RED, la)
        n_died = 0
        n_censored = 0
        median_rows = []
        dead_segments = []
        alive_segments = []
        dead_colors = []
        alive_colors = []
        max_gen_len = 0
        visible_kls: list[float] = []
        # build all dead segments first, then alive segments on top
        for kl, pmv, fork, gl in all_trajs:
            T = min(len(kl), gl)
            max_gen_len = max(max_gen_len, T)
            kl = _rolling_mean(kl[:T], a.roll)
            if a.normalize_kl:
                head = kl[: min(a.calib_tokens, len(kl))]
                head = head[np.isfinite(head)]
                denom = float(np.nanpercentile(head, 95)) if head.size else float("nan")
                if np.isfinite(denom) and denom > 1e-8:
                    kl = kl / denom
                else:
                    # no usable scale (e.g. alpha=0 -> KL==0); skip this traj
                    continue
            visible_kls.extend([float(x) for x in kl if np.isfinite(x)])
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
            ax.add_collection(LineCollection(dead_segments, colors=dead_colors, linewidths=a.line_lw, zorder=1, capstyle="round", joinstyle="round", antialiaseds=True))
        if alive_segments:
            ax.add_collection(LineCollection(alive_segments, colors=alive_colors, linewidths=a.line_lw, zorder=2, capstyle="round", joinstyle="round", antialiaseds=True))
        if median_rows:
            mat = np.asarray(median_rows)
            finite_cols = np.where(np.isfinite(mat).any(axis=0))[0]
            if finite_cols.size:
                med = np.nanmedian(mat[:, : finite_cols[-1] + 1], axis=0)
                med_x = np.arange(len(med))
                if np.nanmax(np.abs(med)) < 1e-6:
                    med = med + 1e-4
                ax.plot(med_x, med, color="black", lw=1.3, alpha=0.9, zorder=4)

        y_hi = _kl_ymax(visible_kls)
        all_visible_kls.extend(visible_kls)
        # calib target: at alpha=1, KL p95 = 1 nat by construction; at general alpha,
        # expected KL ~ alpha^2 (small-step quadratic). Show as horizontal reference.
        # If normalize_kl: each traj is divided by its own p95(KL[:calib_tokens]),
        # so the target collapses to y=1 by construction (independent of alpha).
        if a.normalize_kl:
            target = 1.0
            target_label = rf"calib p95 (tokens<{a.calib_tokens}) = 1"
        else:
            try:
                target = float(alpha) ** 2
            except ValueError:
                target = float("nan")
            target_label = rf"calib target $\alpha^2$={target:.2g}"
        # axhline added unconditionally; off-scale handled later when global ylim is known
        ax.axhline(target, color="black", lw=0.9, ls="--", label=target_label) if np.isfinite(target) and target > 0 else None
        ax.set_title(rf"$\alpha={alpha}$  n={n}  died={n_died}  cens={n_censored}")
        ax.set_xlabel("token t")
        if j == 0:
            ax.set_ylabel("KL / p95(KL[:%d])" % a.calib_tokens if a.normalize_kl else "per-token KL")
        if a.kl_log:
            ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xlim(-1, _panel_xmax(max_gen_len, a.window))
        panel_state.append({"ax": ax, "target": target, "target_label": target_label})
        summary_rows.append({"alpha": alpha, "n": n, "n_died": n_died, "n_censored": n_censored})

    # shared y-axis across all panels for direct cross-alpha comparison
    y_hi_global = _kl_ymax(all_visible_kls)
    for ps in panel_state:
        ax = ps["ax"]
        ax.set_ylim(-y_hi_global * 0.05, y_hi_global)
        target = ps["target"]
        if np.isfinite(target) and target > 0 and target > y_hi_global:
            ax.text(0.98, 0.92, f"{ps['target_label']} off-scale",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.25")


    # legend on first panel only
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=color_alive, lw=2, label=f"alive (pmass >= {a.threshold})"),
        Line2D([0],[0], color=color_dead, lw=2, label=f"dead (pmass < {a.threshold})"),
        Line2D([0],[0], color="black", marker="|", lw=0, label="EOS (right-censored)"),
        Line2D([0],[0], color="black", lw=0.9, ls="--",
               label=(rf"calib p95 (tokens<{a.calib_tokens}) = 1" if a.normalize_kl
                      else r"calib target $\alpha^2$")),
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
