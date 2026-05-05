"""Kaplan-Meier-style survival curves for steered trajectories.

Motivation:
  At alpha=1 KL is near the calibration target by construction (we bisected
  coeff to make it so), so "alpha=1 stays at KL=1" is circular. The honest
  test is whether the model is still COHERENT -- measured here as pmass on
  forced-choice answer tokens. Survival curves separate alphas more cleanly
  than per-token bands.

Death modes (--metric):
  kl     proxy. alive(t) := running_max KL(s) over s<=t < threshold (default 1.0).
         Largely redundant with calibration at alpha=1; read with care.
  pmass  real.  Right-censored Kaplan-Meier on forced-choice mass.
         alive(t) := running_min pmass(s) >= threshold AND rollout reached s.
         Death is irreversible (KM convention).
         Right-censoring: rollouts that EOS'd before fork t drop OUT of the
         denominator at t (they are 'complete', not 'dead'). Uses
         gen_lens_qa[alpha] / gen_lens_eval[alpha] from pmass.json.
         Reads pmass[alpha] (yes/no reasoning prompts). To use the EVAL_PROMPTS
         paired with KL instead, switch the loader to pmass_eval[alpha].

Inputs: outputs/<run>/trajectory.json + pmass.json.

Usage:
  python scripts/survival.py --runs_root outputs_qwen05_w512 \
      --out figs_qwen05_survival --alphas 0.5 1.0 2.0 4.0 \
      --metric pmass --thresholds 0.5 0.8 0.95
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(context="notebook", style="whitegrid", palette="deep", font_scale=0.95)
    plt.rcParams.update({
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
except Exception:
    plt.style.use("ggplot")


@dataclass
class Args:
    runs_root: str = "outputs_qwen05_spaghetti"
    out: str = "figs_qwen05_survival"
    window: int = 50
    alphas: tuple[str, ...] = ("0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "4.0")
    thresholds: tuple[float, ...] = (1.0,)
    model_contains: str = "Qwen2.5-0.5B"
    metric: str = "kl"   # 'kl', 'pmass', or 'pmass_eval' (paired w/ KL prompts)


def _load_kl(root: Path, alpha: str, T: int, model_contains: str) -> np.ndarray:
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        calib = d / "calib.json"; traj_p = d / "trajectory.json"
        if not (calib.exists() and traj_p.exists()):
            continue
        meta = json.loads(calib.read_text())
        if meta.get("window") != T or model_contains not in meta.get("model", ""):
            continue
        traj = json.loads(traj_p.read_text())
        per = traj.get("per_prompt_per_t_kl", {}).get(alpha, [])
        for r in per:
            arr = np.full(T, np.nan)
            arr[: len(r)] = r[: T]
            rows.append(arr)
    return np.array(rows) if rows else np.zeros((0, T))


def _load_pmass(root: Path, alpha: str, T: int, model_contains: str,
                key: str = "pmass",
                ) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Return (N, F) pmass array + fork_points + (N,) gen_lens for right-censoring.
    NaN in the pmass array means the rollout EOS'd before that fork ('complete,
    not dead' -- right-censored). gen_lens is the rollout length T per row.
    key in {'pmass','pmass_eval'} -> uses 'gen_lens_qa' / 'gen_lens_eval'.
    """
    rows = []; gen_lens = []; fork = None
    glen_key = "gen_lens_qa" if key == "pmass" else "gen_lens_eval"
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        calib = d / "calib.json"; pm_p = d / "pmass.json"
        if not (calib.exists() and pm_p.exists()):
            continue
        meta = json.loads(calib.read_text())
        if meta.get("window") != T or model_contains not in meta.get("model", ""):
            continue
        pm = json.loads(pm_p.read_text())
        if not pm.get("computed", True):
            continue
        if fork is None:
            fork = pm["fork_points"]
        per = pm.get(key, {}).get(alpha, [])
        glens = pm.get(glen_key, {}).get(alpha, [])
        for i, r in enumerate(per):
            rows.append(r)
            # fall back to T (window) if gen_lens not saved (legacy outputs)
            gen_lens.append(int(glens[i]) if i < len(glens) else T)
    if not rows:
        return np.zeros((0, len(fork or []))), fork or [], np.zeros(0, dtype=int)
    L = max(len(r) for r in rows)
    arr = np.full((len(rows), L), np.nan)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return arr, fork, np.array(gen_lens, dtype=int)


def survival_kl(K: np.ndarray, threshold: float) -> np.ndarray:
    """S(t) = fraction with running_max(K) < threshold."""
    if K.size == 0:
        return np.zeros(0)
    rmax = np.maximum.accumulate(np.nan_to_num(K, nan=-np.inf), axis=1)
    return (rmax < threshold).mean(axis=0)


def survival_pmass(P: np.ndarray, fork: list[int], gen_lens: np.ndarray,
                   threshold: float) -> np.ndarray:
    """Right-censored Kaplan-Meier on pmass.

    A trajectory is 'at risk' at fork t iff its rollout reached t (gen_len >= t).
    A trajectory 'dies' at fork t if pmass(t) < threshold AND it has not died yet.
    Right-censored trajectories (gen_len < t) drop out of the denominator at t
    (they are 'complete', not 'dead' -- per user 2026-05-05).

    KM estimate: S(t) = prod_{s<=t} (1 - d_s / n_s) where d_s = deaths at s,
    n_s = at-risk just before s. Reduces to "fraction alive" if no censoring.
    """
    if P.size == 0:
        return np.zeros(0)
    N, F = P.shape
    fork_arr = np.array(fork[:F])
    # at_risk[i,j] = True if rollout i reached fork[j] (gen_lens[i] >= fork[j])
    at_risk = gen_lens[:, None] >= fork_arr[None, :]
    # 'dead' event: pmass below threshold (treat NaN as not-dead since we right-censor via at_risk)
    P_filled = np.nan_to_num(P, nan=np.inf)
    died_at = (P_filled < threshold) & at_risk      # (N, F)
    # Make death irreversible: once dead, stays dead at all later forks (where still at risk)
    ever_dead = np.maximum.accumulate(died_at.astype(np.int8), axis=1).astype(bool)
    # KM hazard at each fork: d_s / n_s
    S = np.ones(F)
    s = 1.0
    for j in range(F):
        n_s = int(at_risk[:, j].sum())
        if n_s == 0:
            S[j] = s
            continue
        # new deaths at this fork: ever_dead at j AND not at j-1
        if j == 0:
            d_s = int(ever_dead[:, j].sum())
        else:
            d_s = int((ever_dead[:, j] & ~ever_dead[:, j-1] & at_risk[:, j]).sum())
        s *= (1.0 - d_s / n_s)
        S[j] = s
    return S


def main(a: Args):
    root = Path(a.runs_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_panels = len(a.thresholds)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 3.6),
                             sharey=True, squeeze=False)
    # categorical colors with strong contrast across alpha (avoid viridis dark cluster)
    palette = ["#000000", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#e41a1c", "#f781bf"]
    colors = {alpha: palette[i % len(palette)] for i, alpha in enumerate(a.alphas)}

    rows_summary = []
    for j, thr in enumerate(a.thresholds):
        ax = axes[0, j]
        for alpha in a.alphas:
            if a.metric == "kl":
                K = _load_kl(root, alpha, a.window, a.model_contains)
                if K.size == 0:
                    logger.warning(f"no data alpha={alpha}"); continue
                S = survival_kl(K, thr); xs = np.arange(len(S)); n = K.shape[0]
                xlabel = "token t"
            elif a.metric in ("pmass", "pmass_eval"):
                P, fork, gen_lens = _load_pmass(root, alpha, a.window, a.model_contains, key=a.metric)
                if P.size == 0:
                    logger.warning(f"no {a.metric} alpha={alpha}"); continue
                S = survival_pmass(P, fork, gen_lens, thr)
                xs = np.array(fork[: len(S)]); n = P.shape[0]
                xlabel = "fork token t"
            else:
                raise SystemExit(f"unknown metric {a.metric!r}; use 'kl', 'pmass', or 'pmass_eval'")
            # tiny vertical jitter so overlapping S=1.0 lines remain individually visible
            jitter = 0.004 * (list(a.alphas).index(alpha) - (len(a.alphas) - 1) / 2.0)
            ax.step(xs, S + jitter, where="post", color=colors[alpha], lw=2.2, alpha=0.9,
                    label=rf"$\alpha={alpha}$ (n={n})")
            below_half = np.where(S <= 0.5)[0]
            t50 = int(xs[below_half[0]]) if len(below_half) else None
            rows_summary.append({"metric": a.metric, "threshold": thr, "alpha": alpha,
                                  "n": int(n),
                                  "S_mid": float(S[len(S)//2]),
                                  "S_end": float(S[-1]),
                                  "t_S<=0.5": t50})
        if n_panels > 1:
            ax.set_title(f"threshold = {thr:g}")
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.04, 1.08)
        if j == 0:
            ax.set_ylabel("fraction of trajectories alive")
        # legend outside on the right, never covers the axes
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, frameon=False, title=r"$\alpha$ (n)", title_fontsize=9)
        ax.axvline(20, color="k", ls=":", lw=0.7, alpha=0.5)

    fig.suptitle(f"Survival, {a.model_contains}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_p = out / f"survival_{a.metric}.png"
    fig.savefig(out_p, dpi=160, bbox_inches="tight")
    logger.info(f"survival -> {out_p}")

    from tabulate import tabulate
    md = tabulate(rows_summary, headers="keys", tablefmt="pipe", floatfmt=".3f")
    (out / f"survival_{a.metric}.md").write_text(md)
    logger.info("\n" + md)


if __name__ == "__main__":
    main(tyro.cli(Args))
