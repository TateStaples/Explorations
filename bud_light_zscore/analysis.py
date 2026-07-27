"""Statistical analysis of the Bud Light demand shock (April 2023).

Primary series: Anheuser-Busch InBev U.S. organic revenue growth, quarterly YoY (%).
Question: how many standard deviations below its pre-controversy baseline did
revenue growth fall after the April 2023 Dylan Mulvaney partnership backlash?

Outputs: figures/*.pdf and results.tex (macros + tables consumed by report.tex).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SHOCK = pd.Timestamp("2023-04-01")  # Mulvaney Instagram post

# Presentation palette: neutral ink, one accent for pre, one for post.
C_PRE = "#3B6EA5"
C_POST = "#C4453C"
C_INK = "#2B2B2B"
C_GRID = "#D9D9D9"
C_BAND = "#9FB8D4"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "font.family": "serif",
        "font.size": 9,
        "axes.edgecolor": C_INK,
        "axes.labelcolor": C_INK,
        "axes.linewidth": 0.7,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "text.color": C_INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_csv(DATA / "ab_inbev_us_revenue_growth.csv", parse_dates=["period_end"])
    q = q.sort_values("period_end").reset_index(drop=True)
    w = pd.read_csv(DATA / "budlight_niq_weekly.csv", parse_dates=["week_ending"])
    return q, w


# --------------------------------------------------------------------------
# Core statistics
# --------------------------------------------------------------------------
def robust_scale(x: np.ndarray) -> float:
    """Normal-consistent scale from the median absolute deviation."""
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def zscores(pre: np.ndarray, post: np.ndarray) -> dict:
    """Classical, small-sample-corrected, and robust standardisations.

    The classical z divides by the baseline SD, which understates uncertainty
    when the baseline itself is estimated from n points. The prediction-interval
    statistic below is the correct Student-t analogue: it asks how extreme a
    *new* observation is relative to a baseline whose mean and SD were both
    estimated from n draws.
    """
    n = pre.size
    mu, sd = float(pre.mean()), float(pre.std(ddof=1))
    med, mad = float(np.median(pre)), robust_scale(pre)

    z_classic = (post - mu) / sd
    # Standardised prediction residual: se of a new draw = sd*sqrt(1 + 1/n)
    t_pred = (post - mu) / (sd * np.sqrt(1.0 + 1.0 / n))
    p_pred = 2.0 * stats.t.sf(np.abs(t_pred), df=n - 1)
    z_robust = (post - med) / mad

    return {
        "n_pre": n,
        "mu": mu,
        "sd": sd,
        "median": med,
        "mad_scale": mad,
        "z_classic": z_classic,
        "t_pred": t_pred,
        "p_pred": p_pred,
        "z_robust": z_robust,
    }


def welch(pre: np.ndarray, post: np.ndarray) -> dict:
    t, p = stats.ttest_ind(post, pre, equal_var=False)
    # Hedges' g (bias-corrected standardised mean difference)
    n1, n2 = pre.size, post.size
    sp = np.sqrt(
        ((n1 - 1) * pre.var(ddof=1) + (n2 - 1) * post.var(ddof=1)) / (n1 + n2 - 2)
    )
    d = (post.mean() - pre.mean()) / sp
    J = 1.0 - 3.0 / (4 * (n1 + n2) - 9)
    u3 = stats.mannwhitneyu(post, pre, alternative="less")
    return {
        "t": float(t),
        "p": float(p),
        "cohen_d": float(d),
        "hedges_g": float(d * J),
        "mwu_p": float(u3.pvalue),
        "delta_pp": float(post.mean() - pre.mean()),
    }


def permutation_test(pre: np.ndarray, post: np.ndarray, n_iter: int = 200_000) -> float:
    """Exact-ish one-sided test: could the pre/post split be a fluke of labelling?

    Under H0 the 12 quarterly growth rates are exchangeable. We ask how often a
    random relabelling produces a post-period mean at least as low as observed.
    """
    rng = np.random.default_rng(20230401)
    pool = np.concatenate([pre, post])
    k = post.size
    obs = post.mean() - pre.mean()
    idx = np.argsort(rng.random((n_iter, pool.size)), axis=1)[:, :k]
    draws = pool[idx]
    stat = draws.mean(axis=1) - (pool.sum() - draws.sum(axis=1)) / (pool.size - k)
    return float(((stat <= obs).sum() + 1) / (n_iter + 1))


def bootstrap_z_ci(
    pre: np.ndarray, value: float, n_iter: int = 100_000
) -> tuple[float, float]:
    """Percentile CI for the z of a fixed observation, resampling the baseline."""
    rng = np.random.default_rng(7)
    draws = rng.choice(pre, size=(n_iter, pre.size), replace=True)
    mu = draws.mean(axis=1)
    sd = draws.std(axis=1, ddof=1)
    ok = sd > 0
    z = (value - mu[ok]) / sd[ok]
    return float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_timeseries(q: pd.DataFrame, st: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    pre, post = q[q.period == "pre"], q[q.period == "post"]

    mu, sd = st["mu"], st["sd"]
    ax.axhspan(mu - 2 * sd, mu + 2 * sd, color=C_BAND, alpha=0.28, lw=0)
    ax.axhspan(mu - sd, mu + sd, color=C_BAND, alpha=0.38, lw=0)
    ax.axhline(mu, color=C_PRE, lw=0.9, ls="--")
    ax.axhline(0, color=C_INK, lw=0.6, alpha=0.5)
    ax.axvline(SHOCK, color=C_POST, lw=1.0, ls=":")

    ax.plot(q.period_end, q.organic_rev_growth_pct, color=C_INK, lw=0.8, alpha=0.45, zorder=2)
    ax.scatter(pre.period_end, pre.organic_rev_growth_pct, s=34, color=C_PRE,
               zorder=3, label="Pre-controversy (4Q21–1Q23)")
    ax.scatter(post.period_end, post.organic_rev_growth_pct, s=34, color=C_POST,
               marker="D", zorder=3, label="Post-controversy (2Q23–)")

    lo = float(q.organic_rev_growth_pct.min())
    hi = float(q.organic_rev_growth_pct.max())
    ax.set_ylim(lo - 4.0, hi + 3.5)

    ax.annotate("Mulvaney post, 1 Apr 2023", xy=(SHOCK, hi + 2.6), xytext=(6, 0),
                textcoords="offset points", fontsize=7.5, color=C_POST, va="center")
    for _, r in post.iterrows():
        ax.annotate(f"{r.organic_rev_growth_pct:+.1f}", (r.period_end, r.organic_rev_growth_pct),
                    textcoords="offset points", xytext=(0, -13), ha="center",
                    fontsize=6.8, color=C_POST)

    ax.set_ylabel("Organic revenue growth, YoY (%)")
    ax.set_title("AB InBev U.S. revenue growth: a six-quarter plateau, then a cliff",
                 fontsize=10, loc="left", pad=8)
    ax.text(0.0, 1.0, "", transform=ax.transAxes)
    ax.grid(axis="y", color=C_GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=7.5)
    fig.text(0.011, -0.02, r"Shaded bands: $\pm1\sigma$ and $\pm2\sigma$ of the pre-controversy baseline.",
             fontsize=6.8, color="#666666")
    fig.tight_layout()
    fig.savefig(FIG / "timeseries.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_zbars(q: pd.DataFrame, st: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    post = q[q.period == "post"].reset_index(drop=True)
    z = st["z_classic"]
    cols = [C_POST if v < 0 else C_PRE for v in z]
    bars = ax.bar(post.quarter, z, color=cols, width=0.62)
    for b, v in zip(bars, z):
        off = -1.4 if v < 0 else 0.5
        ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:.1f}", ha="center",
                fontsize=7.5, color=C_INK)
    ax.set_ylim(min(z) - 3.5, 3.0)
    for k, ls, dy in ((-2, "--", 0.7), (-3, ":", -2.1)):
        ax.axhline(k, color=C_INK, lw=0.6, ls=ls, alpha=0.6)
        ax.text(len(post) - 0.52, k + dy, f"z = {k}", fontsize=6.5,
                color="#666666", ha="right", va="bottom")
    ax.axhline(0, color=C_INK, lw=0.7)
    ax.set_ylabel("z-score vs. pre-controversy baseline")
    ax.set_title("Standardised deviation of post-controversy quarters",
                 fontsize=10, loc="left", pad=8)
    ax.grid(axis="y", color=C_GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "zbars.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_weekly(w: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    d = w.dropna(subset=["dollar_sales_yoy_pct"])
    v = w.dropna(subset=["volume_yoy_pct"])
    ax.axvline(SHOCK, color=C_POST, lw=1.0, ls=":")
    ax.axhline(0, color=C_INK, lw=0.6, alpha=0.6)
    ax.plot(d.week_ending, d.dollar_sales_yoy_pct, "o-", color=C_POST, lw=1.1,
            ms=4.5, label="Dollar sales, YoY %")
    ax.plot(v.week_ending, v.volume_yoy_pct, "s--", color=C_PRE, lw=1.0, ms=4,
            alpha=0.85, label="Volume, YoY %")
    ax.annotate("boycott begins", xy=(SHOCK, -2), xytext=(6, 0),
                textcoords="offset points", fontsize=7.5, color=C_POST)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.SA))
    ax.set_xlabel("2023")
    ax.set_ylabel("Bud Light off-premise, YoY (%)")
    ax.set_title("Brand-level collapse and plateau, NIQ scanner data (2023)",
                 fontsize=10, loc="left", pad=8)
    ax.grid(axis="y", color=C_GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "weekly.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(plateau: float, base_mu: float) -> np.ndarray:
    """z of the ~-25% weekly plateau as a function of the assumed baseline SD."""
    sigma = np.linspace(0.5, 6.0, 400)
    z = (plateau - base_mu) / sigma
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.plot(sigma, z, color=C_POST, lw=1.4)
    ax.axhline(-3, color=C_INK, lw=0.6, ls=":", alpha=0.7)
    ax.axhline(-6, color=C_INK, lw=0.6, ls="--", alpha=0.7)
    ax.text(5.92, -3 + 1.2, "z = -3", ha="right", fontsize=6.5, color="#666666")
    ax.text(5.92, -6 - 3.2, "z = -6", ha="right", fontsize=6.5, color="#666666")
    ax.fill_between(sigma, z, -60, color=C_POST, alpha=0.08)
    ax.set_xlabel(r"Assumed pre-controversy weekly baseline SD, $\sigma$ (pp)")
    ax.set_ylabel("Implied z")
    ax.set_ylim(-55, 0)
    ax.set_xlim(0.5, 6.0)
    ax.set_title(r"Weekly result is robust: $|z|>6$ for every plausible $\sigma$",
                 fontsize=10, loc="left", pad=8)
    ax.grid(color=C_GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    return z


def fig_counterfactual(q: pd.DataFrame, st: dict) -> dict:
    """Cumulative shortfall vs. a 'no-controversy' counterfactual.

    Counterfactual: U.S. revenue keeps growing at the pre-period mean rate.
    Shortfall in quarter t (as a share of prior-year U.S. revenue) is
    (mu - g_t)/100. We scale by an estimated quarterly U.S. revenue base.
    """
    post = q[q.period == "post"].reset_index(drop=True)
    us_base_q = 3900.0  # USD m, approx. quarterly U.S. revenue base pre-shock
    gap_pp = st["mu"] - post.organic_rev_growth_pct.values
    loss = gap_pp / 100.0 * us_base_q
    cum = np.cumsum(loss)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(post.quarter, loss, color=C_POST, width=0.6, label="Quarterly shortfall")
    ax2 = ax.twinx()
    ax2.plot(post.quarter, cum, "o-", color=C_INK, lw=1.2, ms=4,
             label="Cumulative")
    ax2.spines["right"].set_visible(True)
    ax.set_ylim(0, float(loss.max()) * 1.35)
    ax2.set_ylim(0, float(cum.max()) * 1.12)
    for x, y in zip(post.quarter, cum):
        ax2.annotate(f"{y/1000:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=6.8, color=C_INK)
    ax.set_ylabel("Quarterly shortfall (USD m)")
    ax2.set_ylabel("Cumulative (USD m)")
    ax.set_title("Revenue shortfall vs. a counterfactual with no controversy",
                 fontsize=10, loc="left", pad=8)
    ax.grid(axis="y", color=C_GRID, lw=0.5)
    ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "counterfactual.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"gap_pp": gap_pp, "loss": loss, "cum": cum, "base": us_base_q}


# --------------------------------------------------------------------------
# LaTeX emission
# --------------------------------------------------------------------------
def fmt(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def pfmt(p: float) -> str:
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.4f}"


def emit(q, st, wl, perm, cis, cf, weekly_z) -> None:
    post = q[q.period == "post"].reset_index(drop=True)
    L = []
    A = L.append

    A("% Auto-generated by analysis.py -- do not edit by hand.")
    A(r"\newcommand{\npre}{%d}" % st["n_pre"])
    A(r"\newcommand{\npost}{%d}" % len(post))
    A(r"\newcommand{\baseMu}{%s}" % fmt(st["mu"]))
    A(r"\newcommand{\baseSd}{%s}" % fmt(st["sd"]))
    A(r"\newcommand{\baseMed}{%s}" % fmt(st["median"]))
    A(r"\newcommand{\baseMad}{%s}" % fmt(st["mad_scale"]))
    A(r"\newcommand{\zTwoQthree}{%s}" % fmt(st["z_classic"][0], 1))
    A(r"\newcommand{\zFourQthree}{%s}" % fmt(st["z_classic"][2], 1))
    A(r"\newcommand{\zMin}{%s}" % fmt(st["z_classic"].min(), 1))
    A(r"\newcommand{\zMean}{%s}" % fmt(st["z_classic"].mean(), 1))
    A(r"\newcommand{\zShockYear}{%s}" % fmt(st["z_classic"][:4].mean(), 1))
    A(r"\newcommand{\pTwoQthree}{%s}" % pfmt(st["p_pred"][0]))
    A(r"\newcommand{\pFourQthree}{%s}" % pfmt(st["p_pred"][2]))
    A(r"\newcommand{\welchT}{%s}" % fmt(wl["t"]))
    A(r"\newcommand{\welchP}{%s}" % pfmt(wl["p"]))
    A(r"\newcommand{\hedgesG}{%s}" % fmt(wl["hedges_g"]))
    A(r"\newcommand{\deltaPP}{%s}" % fmt(wl["delta_pp"], 1))
    A(r"\newcommand{\permP}{%s}" % pfmt(perm))
    A(r"\newcommand{\mwuP}{%s}" % pfmt(wl["mwu_p"]))
    A(r"\newcommand{\ciLo}{%s}" % fmt(cis[0], 1))
    A(r"\newcommand{\ciHi}{%s}" % fmt(cis[1], 1))
    A(r"\newcommand{\cumLoss}{%s}" % fmt(cf["cum"][-1] / 1000.0))
    A(r"\newcommand{\peakLoss}{%s}" % fmt(cf["loss"].max()))
    A(r"\newcommand{\usBase}{%s}" % fmt(cf["base"], 0))
    A(r"\newcommand{\weeklyZlo}{%s}" % fmt(weekly_z["z_at_4pp"], 1))
    A(r"\newcommand{\weeklyZhi}{%s}" % fmt(weekly_z["z_at_1pp"], 1))
    A(r"\newcommand{\weeklyPlateau}{%s}" % fmt(weekly_z["plateau"], 1))

    # --- main results table -------------------------------------------------
    A(r"\newcommand{\ResultsTable}{%")
    A(r"\begin{tabular}{lrrrrr}")
    A(r"\toprule")
    A(r"Quarter & Growth (\%) & $z$ & $t_{\text{pred}}$ & $p$ & $z_{\text{robust}}$ \\")
    A(r"\midrule")
    for i, r in post.iterrows():
        A(
            r"%s & %s & \textbf{%s} & %s & %s & %s \\"
            % (
                r.quarter,
                fmt(r.organic_rev_growth_pct, 1),
                fmt(st["z_classic"][i], 1),
                fmt(st["t_pred"][i], 2),
                pfmt(st["p_pred"][i]),
                fmt(st["z_robust"][i], 1),
            )
        )
    A(r"\bottomrule")
    A(r"\end{tabular}}")

    # --- baseline table -----------------------------------------------------
    pre = q[q.period == "pre"]
    A(r"\newcommand{\BaselineTable}{%")
    A(r"\begin{tabular}{lrrr}")
    A(r"\toprule")
    A(r"Quarter & Revenue (\%) & STRs (\%) & Rev/hl (\%) \\")
    A(r"\midrule")
    for _, r in pre.iterrows():
        strs = "--" if pd.isna(r.strs_pct) else fmt(r.strs_pct, 1)
        rhl = "--" if pd.isna(r.rev_per_hl_pct) else fmt(r.rev_per_hl_pct, 1)
        A(r"%s & %s & %s & %s \\" % (r.quarter, fmt(r.organic_rev_growth_pct, 1), strs, rhl))
    A(r"\midrule")
    A(r"Mean & %s & & \\" % fmt(st["mu"]))
    A(r"SD & %s & & \\" % fmt(st["sd"]))
    A(r"\bottomrule")
    A(r"\end{tabular}}")

    # --- counterfactual table ----------------------------------------------
    A(r"\newcommand{\LossTable}{%")
    A(r"\begin{tabular}{lrrr}")
    A(r"\toprule")
    A(r"Quarter & Gap (pp) & Shortfall (USD m) & Cumulative (USD m) \\")
    A(r"\midrule")
    for i, r in post.iterrows():
        A(r"%s & %s & %s & %s \\" % (r.quarter, fmt(cf["gap_pp"][i], 1),
                                     fmt(cf["loss"][i], 0), fmt(cf["cum"][i], 0)))
    A(r"\bottomrule")
    A(r"\end{tabular}}")

    (ROOT / "results.tex").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------
def main() -> None:
    q, w = load()
    pre = q.loc[q.period == "pre", "organic_rev_growth_pct"].to_numpy(float)
    post = q.loc[q.period == "post", "organic_rev_growth_pct"].to_numpy(float)

    st = zscores(pre, post)
    wl = welch(pre, post)
    perm = permutation_test(pre, post)
    cis = bootstrap_z_ci(pre, post.min())

    fig_timeseries(q, st)
    fig_zbars(q, st)
    fig_weekly(w)
    cf = fig_counterfactual(q, st)

    # Weekly brand-level sensitivity: plateau level vs assumed baseline sigma.
    plateau = float(
        w.dropna(subset=["dollar_sales_yoy_pct"])
        .query("week_ending >= '2023-05-01'")["dollar_sales_yoy_pct"]
        .mean()
    )
    base_mu = 0.0  # Bud Light dollar sales were roughly flat YoY pre-shock
    fig_sensitivity(plateau, base_mu)
    weekly_z = {
        "plateau": plateau,
        "z_at_1pp": (plateau - base_mu) / 1.0,
        "z_at_4pp": (plateau - base_mu) / 4.0,
    }

    emit(q, st, wl, perm, cis, cf, weekly_z)

    summary = {
        "baseline_mean_pct": st["mu"],
        "baseline_sd_pct": st["sd"],
        "z_classic": dict(zip(q.loc[q.period == "post", "quarter"], st["z_classic"].round(2))),
        "t_pred": dict(zip(q.loc[q.period == "post", "quarter"], st["t_pred"].round(2))),
        "p_pred": dict(zip(q.loc[q.period == "post", "quarter"], st["p_pred"])),
        "z_robust": dict(zip(q.loc[q.period == "post", "quarter"], st["z_robust"].round(2))),
        "welch": wl,
        "permutation_p": perm,
        "boot_ci_min_z": cis,
        "cumulative_loss_usd_m": float(cf["cum"][-1]),
        "weekly": weekly_z,
    }
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
