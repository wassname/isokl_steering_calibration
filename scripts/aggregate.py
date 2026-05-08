"""Aggregate per-cell outputs into Figure 1 + headline table.

Layout (default --no-kl-only): 2 rows x N alpha cols.
  Top row: KL(steered || base) per token on EVAL_PROMPTS.
    Band mode (default):  p50 line + p10..p90 band, rolling-{roll} smooth.
    Spaghetti mode:       individual trajectories.
      default coloring:  red if traj ever crosses KL=1, grey otherwise.
      --color-by-pmass:  segments coloured by paired pmass_eval(t) using a
                         red->yellow->green colormap (dead->alive). Requires
                         pmass.json with non-empty pmass_eval (run_cell.py
                         with --compute-pmass).
  Bottom row: forked-answer pmass at fork_points.
    Uses pmass[alpha] (legacy yes/no reasoning prompts). Different prompt set
    than the KL panel above, so the row-to-row link is across cells, not
    paired per-trajectory. For paired analysis see scripts/survival.py with
    --metric pmass on pmass_eval.

Reading caveats:
  - alpha=1 sits near KL=1 on CALIB_PROMPTS by construction. KL on EVAL_PROMPTS
    only tests generalisation, not budget choice -- not an independent test.
  - The honest coherence test is pmass / pmass_eval (real forced-choice mass
    drop), see survival.py for KM-style curves.

Table: per (model, method, window) c_star mean +/- std across seeds.

Usage:
  # smoothed band:
  python scripts/aggregate.py --runs_root outputs --out figs/
  # raw spaghetti:
  python scripts/aggregate.py --runs_root outputs/qwen05_w512 \
      --out figs/qwen05_pretty_raw --spaghetti --roll 1 \
      --alphas 0.5 1.0 2.0 4.0
  # KL spaghetti coloured by paired pmass_eval:
  python scripts/aggregate.py --runs_root outputs/qwen05_w512 \
      --out figs/qwen05_color --spaghetti --color-by-pmass
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import tyro
from loguru import logger
from tabulate import tabulate
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(context="notebook", style="whitegrid", palette="deep", font_scale=0.95)
    plt.rcParams.update({
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.titlesize": 11,
        "figure.dpi": 110,
    })
except Exception:
    plt.style.use("ggplot")


@dataclass
class Args:
    runs_root: str = "outputs"
    out: str = "figs"
    window: int = 512          # only this window enters the figure
    roll: int = 65             # smoothing window for KL trajectory
    alphas: tuple[str, ...] = ("0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "4.0")
    kl_ymax: float = 6.0
    model_contains: str = ""
    kl_only: bool = False
    spaghetti: bool = False    # plot individual trajectories instead of p10..p90 band
    color_by_pmass: bool = False   # color KL spaghetti lines by paired pmass (requires pmass_eval)
    line_alpha: float | None = None  # per-line alpha override; None = auto clip(2.5/n,.08,.35)
    line_lw: float = 0.18      # per-trajectory linewidth; full opacity needs very thin lines
    median_lw: float = 0.75    # median linewidth
    quantile_lines: bool = False  # clean summary: p10/p50/p90 lines, no fill/spaghetti
    mark_t: int = -1           # optional vertical token marker; -1 disables


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean along axis 0 with edge padding so output len == input len."""
    if len(x) < w or w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def load_cells(root: Path, window: int, model_contains: str = "") -> list[dict]:
    cells = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        calib = d / "calib.json"; traj_p = d / "trajectory.json"; pm_p = d / "pmass.json"
        if not (calib.exists() and traj_p.exists() and pm_p.exists()):
            continue
        meta = json.loads(calib.read_text())
        if meta.get("window") != window:
            continue
        if model_contains and model_contains not in meta.get("model", ""):
            continue
        traj = json.loads(traj_p.read_text())
        pm = json.loads(pm_p.read_text())
        # Skip stale outputs that lack per-prompt KL (pre-redesign).
        if "per_prompt_per_t_kl" not in traj:
            logger.warning(f"skipping stale {d.name} (no per_prompt_per_t_kl)")
            continue
        cells.append({"id": d.name, **meta, "traj": traj, "pmass": pm})
    return cells


def _draw_kl_panel(ax, K: np.ndarray, a: Args, P: np.ndarray | None = None) -> None:
    """Draw KL trajectories on ax. spaghetti mode: thin per-trajectory lines.
    color_by_pmass: each segment colored by pmass(t) using a green->red colormap
    (1.0 = lively green, 0.0 = dead red). Otherwise colored by whether traj
    ever crossed KL=1. Smoothing per-line via _rolling_mean(roll). Black median.
    """
    if not K.size:
        return
    finite_cols = np.where(np.isfinite(K).any(axis=0))[0]
    if finite_cols.size == 0:
        return
    K = K[:, : finite_cols[-1] + 1]
    if P is not None and P.size:
        P = P[:, : K.shape[1]]
    xs = np.arange(K.shape[1])
    if a.spaghetti:
        # optional per-line smoothing
        Kp = np.array([_rolling_mean(row, a.roll) for row in K]) if a.roll > 1 else K
        if a.color_by_pmass and P is not None and P.size and P.shape == K.shape:
            # LineCollection per trajectory, color = pmass(t).
            # Draw greenest (alive) first, reddest (dead) last so red sits on top
            # of the green pile -- otherwise the eye loses the dead trajectories.
            from matplotlib.collections import LineCollection
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("alive", ["#c0392b", "#f1c40f", "#27ae60"])  # dead->alive
            order = np.argsort(-np.nanmean(P, axis=1))   # high pmass first, low pmass last
            for traj, pmass_row in zip(Kp[order], P[order]):
                pts = np.column_stack([xs, traj])
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1),
                                    linewidths=a.line_lw, alpha=float(a.line_alpha if a.line_alpha is not None else np.clip(2.5/max(K.shape[0],1), 0.08, 0.35)))
                lc.set_array(pmass_row[:-1])
                ax.add_collection(lc)
            ax.set_xlim(xs[0], xs[-1])
            med = np.nanmedian(Kp, axis=0)
            ax.plot(xs, med, color="k", lw=a.median_lw)
        else:
            crossed = (K > 1.0).any(axis=1)
            for traj in Kp[~crossed]:
                ax.plot(xs, traj, color="0.55", lw=a.line_lw, alpha=0.5)
            for traj in Kp[crossed]:
                ax.plot(xs, traj, color="C3", lw=a.line_lw, alpha=0.5)
            med = np.nanmedian(Kp, axis=0)
            ax.plot(xs, med, color="k", lw=a.median_lw)
            frac = float(crossed.mean())
            ax.text(0.97, 0.97, f"{frac:.0%} cross KL=1",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="C3" if frac > 0.5 else "0.3")
    else:
        p50 = np.nanpercentile(K, 50, axis=0)
        p10 = np.nanpercentile(K, 10, axis=0)
        p90 = np.nanpercentile(K, 90, axis=0)
        p50s = _rolling_mean(p50, a.roll)
        p10s = _rolling_mean(p10, a.roll)
        p90s = _rolling_mean(p90, a.roll)
        if a.quantile_lines:
            # standard quantile fan: outer band light, inner band dark, p50 line
            p25s = _rolling_mean(np.nanpercentile(K, 25, axis=0), a.roll)
            p75s = _rolling_mean(np.nanpercentile(K, 75, axis=0), a.roll)
            ax.fill_between(xs, p10s, p90s, alpha=0.15, color="C0", lw=0, label="p10..p90")
            ax.fill_between(xs, p25s, p75s, alpha=0.32, color="C0", lw=0, label="p25..p75")
            ax.plot(xs, p50s, color="C0", lw=1.5, label="p50")
        else:
            ax.fill_between(xs, p10s, p90s, alpha=0.25, color="C0", lw=0)
            ax.plot(xs, p50s, color="C0", lw=1.6)


def make_kl_figure(cells: list[dict], a: Args, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(a.alphas), figsize=(3.7 * len(a.alphas), 3.4),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    label = a.model_contains or "all models"
    n_max = max((_pool_kl(cells, alpha, T=a.window).shape[0] for alpha in a.alphas), default=0)
    mode = f"individual trajectories, roll={a.roll}, color=pmass" if (a.spaghetti and a.color_by_pmass) \
        else "individual trajectories (red=ever crossed KL=1)" if a.spaghetti \
        else f"shaded quantiles p10/p25/p50/p75/p90, roll={a.roll}" if a.quantile_lines \
        else f"p50 + p10..p90 band, smoothed rolling-{a.roll}"
    fig.suptitle(
        f"KL trajectory on N={n_max} held-out long-form prompts ({label})\n"
        f"{mode}. Solid horizontal: KL=1 nat.",
        fontsize=10,
    )

    x_stop = 1
    y_data: list[float] = []
    for alpha in a.alphas:
        K = _pool_kl(cells, alpha, T=a.window)
        finite_cols = np.where(np.isfinite(K).any(axis=0))[0]
        if finite_cols.size:
            x_stop = max(x_stop, int(finite_cols[-1] + 1))
            vals = K[:, : finite_cols[-1] + 1]
            y_data.extend([float(x) for x in vals.ravel() if np.isfinite(x) and x >= 0.0])
    x_max = float(min(a.window, max(20, int(x_stop * 1.05))))
    if y_data:
        y_max = float(min(a.kl_ymax, max(1e-3, np.nanpercentile(np.asarray(y_data), 99) * 1.4)))
    else:
        y_max = 1.1

    for j, alpha in enumerate(a.alphas):
        ax = axes[0, j]
        K = _pool_kl(cells, alpha, T=a.window)
        Pe = _pool_pmass_eval(cells, alpha, _first_fork(cells), T=a.window) if a.color_by_pmass else None
        _draw_kl_panel(ax, K, a, P=Pe)
        if y_max >= 1.0:
            ax.axhline(1.0, color="k", lw=0.7)
        else:
            ax.text(0.98, 0.92, "KL=1 off-scale", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="0.25")
        if a.mark_t >= 0:
            ax.axvline(a.mark_t, color="k", ls=":", lw=0.6)
        ax.set_title(rf"$\alpha = {alpha}$  (n={K.shape[0]} traj)")
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("token")
        if j == 0:
            ax.set_ylabel("KL")

    if a.color_by_pmass:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        cmap = LinearSegmentedColormap.from_list("alive", ["#c0392b", "#f1c40f", "#27ae60"])
        sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap); sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[0, :].tolist(), location="right", shrink=0.72, pad=0.015, fraction=0.012)
        cbar.set_label("pmass", labelpad=2)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if a.quantile_lines:
        q_path = out_path.with_name(out_path.stem + "_quantile_lines" + out_path.suffix)
        fig.savefig(q_path, dpi=160, bbox_inches="tight")
        logger.info(f"KL quantile-line figure -> {q_path}")
    logger.info(f"KL-only figure -> {out_path}")


def make_table(root: Path) -> pl.DataFrame:
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        calib = d / "calib.json"
        if not calib.exists():
            continue
        rows.append(json.loads(calib.read_text()) | {"run": d.name})
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("model").str.split("/").list.last().alias("model_short"))
    g = (df.group_by(["model_short", "method", "window"])
           .agg(pl.col("c_star").mean().alias("c_mean"),
                pl.col("c_star").std().alias("c_std"),
                pl.len().alias("n_seeds"))
           .sort(["model_short", "method", "window"]))
    g = g.with_columns(
        (pl.col("c_std") / pl.col("c_mean").abs()).alias("c_cv"),
    )
    return g


def _pool_kl(cells: list[dict], alpha: str, T: int) -> np.ndarray:
    """Stack per-prompt KL trajectories from all cells -> (N, T) ndarray."""
    rows = []
    for c in cells:
        per_prompt = c["traj"]["per_prompt_per_t_kl"].get(alpha, [])
        for r in per_prompt:
            arr = np.full(T, np.nan)
            arr[: len(r)] = r[: T]
            rows.append(arr)
    return np.array(rows) if rows else np.zeros((0, T))


def _first_fork(cells: list[dict]) -> list[int]:
    for c in cells:
        if c["pmass"].get("computed", True):
            return c["pmass"]["fork_points"]
    return []


def _pool_pmass_eval(cells: list[dict], alpha: str, fork_points: list[int], T: int) -> np.ndarray:
    """Pool pmass on EVAL_PROMPTS (paired with KL) and interpolate from fork
    points to per-token, returning (N, T). NaN for cells without pmass_eval.
    """
    rows = []
    for c in cells:
        if not c["pmass"].get("computed", True):
            continue
        per_prompt = c["pmass"].get("pmass_eval", {}).get(alpha, [])
        for r in per_prompt:
            xs = np.array(fork_points[: len(r)])
            ys = np.array(r, dtype=float)
            t = np.arange(T)
            interp = np.interp(t, xs, ys, left=ys[0], right=ys[-1])
            rows.append(interp)
    return np.array(rows) if rows else np.zeros((0, T))


def _pool_pmass(cells: list[dict], alpha: str) -> tuple[np.ndarray, list[int]]:
    rows = []
    fork = None
    for c in cells:
        if not c["pmass"].get("computed", True):
            continue
        f = c["pmass"]["fork_points"]
        if fork is None: fork = f
        per_prompt = c["pmass"]["pmass"].get(alpha, [])
        for r in per_prompt:
            rows.append(r)
    if not rows:
        return np.zeros((0, len(fork or []))), fork or []
    L = max(len(r) for r in rows)
    arr = np.full((len(rows), L), np.nan)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return arr, fork


def make_figure(cells: list[dict], a: Args, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(a.alphas), figsize=(4.0 * len(a.alphas), 5.5),
                             sharex="row", sharey="row", squeeze=False)
    n_cells = len(cells)
    fig.suptitle(
        f"Figure 1: iso-KL calibration on Qwen2.5-0.5B-Instruct\n"
        f"Top: KL(steered || base) per token on long-form held-out prompts.\n"
        f"Bottom: forked-answer pmass on yes/no reasoning prompts (high=alive, low=collapsed).\n"
        f"{n_cells} cells (3 methods x 1 seed) x N=8 prompts. "
        f"Solid h-line: KL=1 nat. Dotted v-line: t=20.",
        fontsize=10,
    )

    for j, alpha in enumerate(a.alphas):
        # ---- top: KL ----
        ax = axes[0, j]
        K = _pool_kl(cells, alpha, T=a.window)
        Pe = _pool_pmass_eval(cells, alpha, _first_fork(cells), T=a.window) if a.color_by_pmass else None
        _draw_kl_panel(ax, K, a, P=Pe)
        ax.axhline(1.0, color="k", lw=1.0)
        ax.axvline(20, color="k", ls=":", lw=0.8)
        ax.set_title(rf"$\alpha = {alpha}$  (n={K.shape[0]} traj)")
        ax.set_ylim(0, a.kl_ymax)
        if j == 0:
            ax.set_ylabel("KL(steered || base)  [nats]")
        ax.set_xlabel("token")

        # ---- bottom: pmass ----
        ax2 = axes[1, j]
        P, fork = _pool_pmass(cells, alpha)
        if P.size:
            xs = np.array(fork[: P.shape[1]])
            if a.spaghetti:
                for row in P:
                    ax2.plot(xs, row, color="C1", lw=0.6, alpha=0.5)
                ax2.plot(xs, np.nanmedian(P, axis=0), color="k", lw=1.6, marker="o", ms=3)
            else:
                p50 = np.nanpercentile(P, 50, axis=0)
                p10 = np.nanpercentile(P, 10, axis=0)
                p90 = np.nanpercentile(P, 90, axis=0)
                ax2.fill_between(xs, p10, p90, alpha=0.25, color="C1", lw=0)
                ax2.plot(xs, p50, color="C1", lw=1.6, marker="o", ms=3)
        ax2.axvline(20, color="k", ls=":", lw=0.8)
        ax2.set_ylim(-0.02, 1.05)
        if j == 0:
            ax2.set_ylabel('pmass = p(true/1) + p(false/0)\nat fork t')
        ax2.set_xlabel("fork token")

    if a.color_by_pmass:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        cmap = LinearSegmentedColormap.from_list("alive", ["#c0392b", "#f1c40f", "#27ae60"])
        sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap); sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[0, :].tolist(), location="right", shrink=0.8, pad=0.02)
        cbar.set_label("pmass (0=dead, 1=alive)")

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    logger.info(f"figure -> {out_path}")


def main(a: Args):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.runs_root)
    cells = load_cells(root, window=a.window, model_contains=a.model_contains)
    if not cells:
        raise SystemExit(f"no cells with window={a.window} under {root}")
    logger.info(f"loaded {len(cells)} cells (window={a.window})")

    df = make_table(root)
    if not df.is_empty():
        df.write_csv(out / "table.csv")
        md = tabulate(df.rows(), headers=df.columns, tablefmt="pipe", floatfmt=".3f")
        (out / "table.md").write_text(md)
        logger.info(f"table -> {out/'table.md'}\n{md}")

    if a.kl_only:
        make_kl_figure(cells, a, out / "figure1_kl_only.png")
    else:
        make_figure(cells, a, out / "figure1.png")


if __name__ == "__main__":
    main(tyro.cli(Args))
