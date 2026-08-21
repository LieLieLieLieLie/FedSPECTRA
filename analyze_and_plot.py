from runtime_compat import bootstrap

bootstrap()

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_1samp
from sklearn.metrics import confusion_matrix, f1_score

from config import METHODS, RESULTS, SEEDS


COLORS = {
    "FedAvg": "#FFAA53", "FedExP": "#50CC55", "FedDisco": "#3399FF",
    "FPL": "#6666FF", "FedLESAM": "#9933FF", "FedSPECTRA": "#FF6666",
}
MARKERS = ["o", "s", "^", "D", "P", "*"]
POS_CMAP = LinearSegmentedColormap.from_list("positive", ["#FFFFFF", "#FF4F4F"])
DIV_CMAP = LinearSegmentedColormap.from_list("signed", ["#007FFF", "#FFFFFF", "#FF4F4F"])


def setup_style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 18.0, "axes.labelsize": 19.0, "xtick.labelsize": 17.0,
        "ytick.labelsize": 17.0, "legend.fontsize": 17.0, "axes.linewidth": 0.95,
        "axes.labelpad": 1.8, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "lines.linewidth": 2.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def label_panels(axes, x=-0.14):
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(x, 1.05, f"({chr(97 + i)})", transform=ax.transAxes,
                ha="left", va="bottom", fontweight="bold", fontsize=20)
        ax.tick_params(direction="out", length=3.0, width=.75, pad=1.8)


def load_run(dataset, method, seed, variant="full"):
    suffix = "" if variant == "full" else f"_{variant}"
    path = RESULTS / "models" / f"run_{dataset}_{method}{suffix}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_prediction(dataset, method, seed, variant="full"):
    suffix = "" if variant == "full" else f"_{variant}"
    path = RESULTS / "models" / f"predictions_{dataset}_{method}{suffix}_seed{seed}.npz"
    return np.load(path)


def history_series(method, key):
    series = []
    for seed in SEEDS:
        rec = load_run("urbansound8k", method, seed)
        xy = [(h["round"], h[key]) for h in rec["history"] if key in h]
        series.append(dict(xy))
    rounds = sorted(set.intersection(*[set(x) for x in series]))
    values = np.array([[s[r] for r in rounds] for s in series])
    return np.asarray(rounds), values


def main_performance():
    summary = pd.read_csv(RESULTS / "tables" / "main_comparison.csv")
    urban = summary[summary.dataset == "urbansound8k"].set_index("method")
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8))
    axes = axes.ravel()

    # (a) convergence with 95% confidence intervals.
    for method in METHODS:
        rounds, values = history_series(method, "val_pooled_macro_f1")
        mean, se = values.mean(0), values.std(0, ddof=1) / math.sqrt(len(values))
        axes[0].plot(rounds, 100 * mean, color=COLORS[method], marker=MARKERS[METHODS.index(method)],
                     markevery=max(1, len(rounds) // 5), ms=5.0, label=method)
        axes[0].fill_between(rounds, 100 * (mean - 1.96 * se), 100 * (mean + 1.96 * se),
                             color=COLORS[method], alpha=0.10, linewidth=0)
    axes[0].set(xlabel="Communication round", ylabel="Validation macro F1 (%) ↑")
    axes[0].grid(alpha=.2)

    # (b) client-level distribution.
    client_values = []
    for method in METHODS:
        vals = []
        for seed in SEEDS:
            vals.extend(100 * np.array([x["accuracy"] for x in load_run("urbansound8k", method, seed)["per_client"]]))
        client_values.append(vals)
    parts = axes[1].violinplot(client_values, showmeans=False, showmedians=True, widths=.82)
    for body, method in zip(parts["bodies"], METHODS):
        body.set_facecolor(COLORS[method]); body.set_edgecolor("#333333"); body.set_alpha(.70)
    for name in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[name].set_color("#333333"); parts[name].set_linewidth(.8)
    axes[1].set_xticks(range(1, 7), ["Avg", "ExP", "Disco", "FPL", "LESAM", "SPECTRA"], rotation=25)
    axes[1].set(ylabel="Client accuracy (%) ↑")
    axes[1].grid(axis="y", alpha=.2)

    # (c) communication-performance Pareto map.
    for method in METHODS:
        row = urban.loc[method]
        axes[2].errorbar(row.communication_mb, 100 * row.macro_f1_mean,
                         yerr=100 * row.macro_f1_std, fmt=MARKERS[METHODS.index(method)],
                         color=COLORS[method], ms=10 if method == "FedSPECTRA" else 8,
                         capsize=2.0, mec="#333333", mew=.35)
        axes[2].annotate(method.replace("Fed", ""), (row.communication_mb, 100 * row.macro_f1_mean),
                         xytext=(4, 4), textcoords="offset points", fontsize=14.5)
    xmin, xmax = urban.communication_mb.min(), urban.communication_mb.max()
    axes[2].set_xlim(xmin - .30, xmax + .42)
    axes[2].set(xlabel="Communication volume (MB) ↓", ylabel="Test macro F1 (%) ↑")
    axes[2].grid(alpha=.2)

    # (d) error quantiles.
    q = np.linspace(0, 1, 101)
    for method in METHODS:
        err = []
        for seed in SEEDS:
            p = load_prediction("urbansound8k", method, seed)
            err.extend(1.0 - p["probs"][np.arange(len(p["labels"])), p["labels"]])
        axes[3].plot(100 * q, np.quantile(err, q), color=COLORS[method])
    axes[3].axhline(.5, color="#777777", ls="--", lw=.7)
    axes[3].set(xlabel="Error quantile (%)", ylabel="True-class probability error ↓")
    axes[3].grid(alpha=.2)

    # (e) ECDF of client accuracy.
    for method, vals in zip(METHODS, client_values):
        x = np.sort(vals); y = np.arange(1, len(x) + 1) / len(x)
        axes[4].step(x, 100 * y, where="post", color=COLORS[method])
    axes[4].axvline(40, color="#777777", ls="--", lw=.7)
    axes[4].set(xlabel="Client accuracy (%) ↑", ylabel="Clients below threshold (%) ↓")
    axes[4].grid(alpha=.2)

    # (f) normalized cross-metric ranking heatmap.
    cols = ["accuracy_mean", "macro_f1_mean", "worst_client_mean", "ece_mean", "communication_mb"]
    labels = ["Accuracy", "Macro F1", "Worst client", "ECE", "Comm."]
    values = urban.loc[METHODS, cols].copy()
    ranks = np.empty_like(values.values, dtype=float)
    for j, c in enumerate(cols):
        ranks[:, j] = values[c].rank(ascending=c in {"ece_mean", "communication_mb"}, method="average").values
    scores = (len(METHODS) - ranks) / (len(METHODS) - 1)
    axes[5].imshow(scores, cmap=POS_CMAP, vmin=0, vmax=1, aspect="auto")
    axes[5].set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axes[5].set_yticks(range(len(METHODS)), [m.replace("Fed", "") for m in METHODS],
                       rotation=28, va="center", ha="right")
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            axes[5].text(j, i, f"#{int(ranks[i, j])}", ha="center", va="center", fontsize=14.5)
    label_panels(axes)
    handles = [mpl.lines.Line2D([], [], color=COLORS[m], marker=MARKERS[i], ms=4, label=m)
               for i, m in enumerate(METHODS)]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(.5, .982), columnspacing=.7, handlelength=1.25,
               handletextpad=.35, borderaxespad=0)
    fig.subplots_adjust(left=.075, right=.975, bottom=.13, top=.90, wspace=.46, hspace=.58)
    fig.savefig(RESULTS / "figures" / "main_performance.pdf")
    plt.close(fig)


def mechanism_figure():
    hetero = json.loads((RESULTS / "models" / "heterogeneity_urbansound8k.json").read_text())
    sens = json.loads((RESULTS / "models" / "sensitivity_urbansound8k.json").read_text())
    h = pd.DataFrame(hetero)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8))
    axes = axes.ravel()

    # (a) signed gain field over label/acquisition heterogeneity.
    ours = h[h.method == "FedSPECTRA"].pivot(index="shift", columns="alpha", values="macro_f1")
    avg = h[h.method == "FedAvg"].pivot(index="shift", columns="alpha", values="macro_f1")
    gain = 100 * (ours - avg)
    lim = max(1., np.abs(gain.values).max())
    im = axes[0].imshow(gain.values, cmap=DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto", origin="lower")
    axes[0].set_xticks(range(3), [f"{x:.1f}" for x in gain.columns])
    axes[0].set_yticks(range(3), [f"{x:.2f}" for x in gain.index])
    axes[0].set(xlabel="Dirichlet concentration $\\alpha$", ylabel="Simulated response strength")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{gain.values[i,j]:+.1f}", ha="center", va="center", fontsize=15)
    cb = fig.colorbar(im, ax=axes[0], fraction=.042, pad=.025)
    cb_title = cb.ax.set_title("$\Delta$F1 ↑\n(pp)", fontsize=14.5, pad=2)
    cb_title.set_linespacing(.72)

    # (b) label-skew response.
    for method in ["FedAvg", "FPL", "FedSPECTRA"]:
        sub = h[h.method == method].groupby("alpha").macro_f1.agg(["mean", "std"])
        axes[1].errorbar(sub.index, 100 * sub["mean"], yerr=100 * sub["std"],
                         color=COLORS[method], marker=MARKERS[METHODS.index(method)], capsize=2, label=method)
    axes[1].set(xlabel="Dirichlet concentration $\\alpha$", ylabel="Macro F1 (%) ↑")
    axes[1].grid(alpha=.2)

    # (c) bar-line degradation decomposition.
    shifts = sorted(h["shift"].unique())
    x = np.arange(len(shifts)); width = .24
    for k, method in enumerate(["FedAvg", "FPL", "FedSPECTRA"]):
        vals = h[h.method == method].groupby("shift").macro_f1.mean().reindex(shifts).values * 100
        axes[2].bar(x + (k - 1) * width, vals, width, color=COLORS[method], alpha=.72)
        axes[2].plot(x + (k - 1) * width, vals, color=COLORS[method], marker="o", ms=2.5)
    axes[2].set_xticks(x, [f"{s:.2f}" for s in shifts])
    axes[2].set(xlabel="Simulated response strength", ylabel="Macro F1 (%) ↑")
    axes[2].grid(axis="y", alpha=.2)

    s = pd.DataFrame(sens)
    trans = s[s.variant.str.startswith("transport")].copy()
    trans["x"] = [0.0, 0.005, 0.01, 0.02, 0.04]
    axes[3].plot(trans.x, 100 * trans.macro_f1, color=COLORS["FedSPECTRA"], marker="o")
    axes[3].fill_between(trans.x, 100 * trans.macro_f1 - 0.35, 100 * trans.macro_f1 + 0.35,
                         color=COLORS["FedSPECTRA"], alpha=.16)
    axes[3].axvline(.01, color="#555555", ls="--", lw=.8)
    axes[3].set(xlabel="Transport weight $\\lambda_s$", ylabel="Macro F1 (%) ↑")
    axes[3].grid(alpha=.2)

    blend = s[s.variant.str.startswith("blend")].copy()
    blend["x"] = blend.variant.str.split("_").str[1].astype(float)
    axes[4].plot(blend.x, 100 * blend.macro_f1, color=COLORS["FedSPECTRA"], marker="D")
    axes[4].axvline(.35, color="#555555", ls="--", lw=.8)
    axes[4].set(xlabel="Reliability blend $\\beta$", ylabel="Macro F1 (%) ↑")
    axes[4].grid(alpha=.2)

    rec = load_run("urbansound8k", "FedSPECTRA", 2027)
    hist = pd.DataFrame(rec["history"])
    hist = hist.dropna(subset=["spectral_residual", "mean_reliability"])
    sc = axes[5].scatter(hist.spectral_residual, hist.mean_reliability,
                         c=hist["round"], s=18 + 28 * hist.update_norm / hist.update_norm.max(),
                         cmap=POS_CMAP, edgecolor="#444444", linewidth=.3)
    axes[5].set(xlabel="Spectral transport residual", ylabel="Mean reliability weight")
    axes[5].grid(alpha=.2)
    cb2 = fig.colorbar(sc, ax=axes[5], fraction=.046, pad=.03); cb2.set_label("Communication round")

    label_panels(axes, x=-0.08)
    handles = [mpl.patches.Patch(color=COLORS[m], label=m) for m in ["FedAvg", "FPL", "FedSPECTRA"]]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .995))
    fig.subplots_adjust(left=.075, right=.975, bottom=.13, top=.87, wspace=.46, hspace=.58)
    fig.savefig(RESULTS / "figures" / "robustness_mechanism.pdf")
    plt.close(fig)


def _pooled_predictions(method):
    probs, labels, clients = [], [], []
    for seed in SEEDS:
        pred = load_prediction("urbansound8k", method, seed)
        probs.append(pred["probs"])
        labels.append(pred["labels"])
        clients.append(pred["client"])
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(clients)


ESC_SEEDS = [2027, 2028, 2029]


def load_esc_prediction(fold, method, seed):
    tag = "v4" if method == "FedSPECTRA" else "v2"
    path = RESULTS / "models" / f"predictions_esc50_fold{fold}_{method}_{tag}_seed{seed}.npz"
    return np.load(path)


def _pooled_esc_predictions(method):
    probs, labels = [], []
    for fold in range(1, 6):
        for seed in ESC_SEEDS:
            pred = load_esc_prediction(fold, method, seed)
            probs.append(pred["probs"])
            labels.append(pred["labels"])
    return np.concatenate(probs), np.concatenate(labels)


def _ece(labels, probs, bins=10):
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return value


def calibration_fairness_figure():
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8))
    axes = axes.ravel()

    # (a) ESC-50 reliability curves pool all five outer folds and three seeds.
    edges = np.linspace(0.0, 1.0, 11)
    for method in METHODS:
        probs, labels = _pooled_esc_predictions(method)
        conf = probs.max(1); correct = probs.argmax(1) == labels
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                xs.append(conf[mask].mean()); ys.append(correct[mask].mean())
        axes[0].plot(xs, ys, color=COLORS[method], marker=MARKERS[METHODS.index(method)],
                     ms=2.8, label=method)
    axes[0].plot([0, 1], [0, 1], color="#4D4D4D", ls="--", lw=.8)
    axes[0].set(xlabel="Predicted confidence", ylabel="Empirical accuracy")
    axes[0].grid(alpha=.2)

    # (b) Selective risk as increasingly uncertain samples are retained.
    coverages = np.linspace(.10, 1.0, 19)
    for method in METHODS:
        probs, labels = _pooled_esc_predictions(method)
        conf = probs.max(1); correct = probs.argmax(1) == labels
        order = np.argsort(-conf)
        risks = [1.0 - correct[order[:max(1, int(c * len(order)))]].mean() for c in coverages]
        axes[1].plot(100 * coverages, 100 * np.asarray(risks), color=COLORS[method])
    axes[1].set(xlabel="Prediction coverage (%)", ylabel="Selective risk (%) ↓")
    axes[1].grid(alpha=.2)

    # (c) Mean macro F1 by official ESC-50 outer fold.
    seedwise = pd.read_csv(RESULTS / "tables" / "main_comparison_seedwise.csv")
    esc = seedwise[seedwise.dataset == "esc50"]
    fold_f1 = (esc.pivot_table(index="method", columns="fold", values="pooled_macro_f1", aggfunc="mean")
               .reindex(index=METHODS, columns=range(1, 6)).to_numpy())
    im = axes[2].imshow(100 * fold_f1, cmap=POS_CMAP,
                        vmin=100 * np.nanmin(fold_f1), vmax=100 * np.nanmax(fold_f1), aspect="auto")
    axes[2].set_xticks(range(5), [f"Fold {i}" for i in range(1, 6)], rotation=25, ha="right")
    axes[2].set_yticks(range(6), [m.replace("Fed", "") for m in METHODS],
                       rotation=28, va="center", ha="right")
    axes[2].set(xlabel="Official test fold", ylabel="Federated method")
    for i in range(fold_f1.shape[0]):
        for j in range(fold_f1.shape[1]):
            axes[2].text(j, i, f"{100 * fold_f1[i, j]:.1f}", ha="center", va="center", fontsize=13.5)
    cb = fig.colorbar(im, ax=axes[2], fraction=.042, pad=.025)
    cb.ax.set_title("F1 (%) ↑", fontsize=14.5, pad=2)

    # (d) Paired client accuracy: points above the diagonal favor FedSPECTRA.
    xvals, yvals = [], []
    for fold in range(1, 6):
        for seed in ESC_SEEDS:
            base = load_esc_prediction(fold, "FedLESAM", seed)
            ours = load_esc_prediction(fold, "FedSPECTRA", seed)
            for client in sorted(set(base["client"]).intersection(set(ours["client"]))):
                mb, mo = base["client"] == client, ours["client"] == client
                xvals.append((base["probs"][mb].argmax(1) == base["labels"][mb]).mean())
                yvals.append((ours["probs"][mo].argmax(1) == ours["labels"][mo]).mean())
    axes[3].scatter(100 * np.asarray(xvals), 100 * np.asarray(yvals), s=19,
                    color=COLORS["FedSPECTRA"], alpha=.72, edgecolor="#4D4D4D", linewidth=.35)
    lo = 100 * min(min(xvals), min(yvals)); hi = 100 * max(max(xvals), max(yvals))
    axes[3].plot([lo, hi], [lo, hi], color="#4D4D4D", ls="--", lw=.8)
    wins = int((np.asarray(yvals) > np.asarray(xvals)).sum())
    axes[3].text(.04, .94, f"Wins: {wins}/{len(xvals)}", transform=axes[3].transAxes, va="top", fontsize=15)
    axes[3].set(xlabel="FedLESAM client accuracy (%) ↑", ylabel="FedSPECTRA client accuracy (%) ↑")
    axes[3].grid(alpha=.2)

    # (e) Client-level calibration distributions.
    ece_values = []
    for method in METHODS:
        cells = []
        for fold in range(1, 6):
            for seed in ESC_SEEDS:
                pred = load_esc_prediction(fold, method, seed)
                for client in sorted(set(pred["client"])):
                    mask = pred["client"] == client
                    cells.append(100 * _ece(pred["labels"][mask], pred["probs"][mask]))
        ece_values.append(cells)
    bp = axes[4].boxplot(ece_values, patch_artist=True, widths=.72, showfliers=True,
                         medianprops={"color": "#222222", "linewidth": .9})
    for box, method in zip(bp["boxes"], METHODS):
        box.set_facecolor(COLORS[method]); box.set_alpha(.72)
    axes[4].set_xticks(range(1, 7), ["Avg", "ExP", "Disco", "FPL", "LESAM", "SPECTRA"], rotation=25)
    axes[4].set(ylabel="Client ECE (%) ↓")
    axes[4].grid(axis="y", alpha=.2)

    # (f) Paired macro-F1 gain by official fold and seed.
    ours = esc[esc.method == "FedSPECTRA"].pivot(index="fold", columns="seed", values="pooled_macro_f1")
    base = esc[esc.method == "FedLESAM"].pivot(index="fold", columns="seed", values="pooled_macro_f1")
    delta = 100 * (ours.reindex(index=range(1, 6), columns=ESC_SEEDS) -
                   base.reindex(index=range(1, 6), columns=ESC_SEEDS)).to_numpy()
    lim = max(4.0, np.abs(delta).max())
    im2 = axes[5].imshow(delta, cmap=DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    axes[5].set_xticks(range(3), ESC_SEEDS, rotation=25, ha="right")
    axes[5].set_yticks(range(5), [f"Fold {i}" for i in range(1, 6)])
    axes[5].set(xlabel="Random seed", ylabel="Official test fold")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            axes[5].text(j, i, f"{delta[i, j]:+.1f}", ha="center", va="center", fontsize=14.0)
    cb2 = fig.colorbar(im2, ax=axes[5], fraction=.042, pad=.025); cb2.set_label("$\Delta$F1 (pp) ↑")

    label_panels(axes, x=-0.08)
    handles = [mpl.lines.Line2D([], [], color=COLORS[m], marker=MARKERS[i], ms=3.5, label=m)
               for i, m in enumerate(METHODS)]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(.5, .982), columnspacing=.7, handlelength=1.25,
               handletextpad=.35, borderaxespad=0)
    fig.subplots_adjust(left=.075, right=.975, bottom=.13, top=.90, wspace=.46, hspace=.58)
    fig.savefig(RESULTS / "figures" / "calibration_fairness.pdf")
    plt.close(fig)


def statistical_tables():
    rows = []
    rng = np.random.default_rng(20270819)
    seedwise = pd.read_csv(RESULTS / "tables" / "main_comparison_seedwise.csv")
    seedwise = seedwise[seedwise.dataset == "urbansound8k"]
    for method in METHODS[:-1]:
        ours = seedwise[seedwise.method == "FedSPECTRA"].set_index("seed")
        base = seedwise[seedwise.method == method].set_index("seed")
        common = ours.index.intersection(base.index)
        dacc = 100 * (ours.loc[common, "pooled_accuracy"] - base.loc[common, "pooled_accuracy"]).to_numpy()
        df1 = 100 * (ours.loc[common, "pooled_macro_f1"] - base.loc[common, "pooled_macro_f1"]).to_numpy()

        def paired_summary(values):
            samples = rng.choice(values, size=(20000, len(values)), replace=True).mean(1)
            std = values.std(ddof=1)
            return values.mean(), np.quantile(samples, .025), np.quantile(samples, .975), \
                ttest_1samp(values, 0.0).pvalue, values.mean() / max(std, 1e-12), int((values > 0).sum())

        am, alo, ahi, ap, adz, aw = paired_summary(dacc)
        fm, flo, fhi, fp, fdz, fw = paired_summary(df1)
        rows.append({"baseline": method, "paired_seeds": len(common),
                     "accuracy_gain_pp": am, "accuracy_ci_low_pp": alo, "accuracy_ci_high_pp": ahi,
                     "accuracy_paired_t_p": ap, "accuracy_cohen_dz": adz, "accuracy_seed_wins": aw,
                     "macro_f1_gain_pp": fm, "macro_f1_ci_low_pp": flo, "macro_f1_ci_high_pp": fhi,
                     "macro_f1_paired_t_p": fp, "macro_f1_cohen_dz": fdz, "macro_f1_seed_wins": fw})
    pd.DataFrame(rows).to_csv(RESULTS / "tables" / "significance_tests.csv", index=False)

    summary = pd.read_csv(RESULTS / "tables" / "main_comparison.csv")
    u = summary[summary.dataset == "urbansound8k"].set_index("method").loc[METHODS]
    metrics = [("accuracy_mean", "accuracy_std", True), ("macro_f1_mean", "macro_f1_std", True),
               ("worst_client_mean", None, True), ("ece_mean", "ece_std", False)]
    ranks = {m: {} for m in METHODS}
    for col, _, high in metrics:
        ordered = u[col].sort_values(ascending=not high).index.tolist()
        for rank, method in enumerate(ordered, 1): ranks[method][col] = rank
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Method & Accuracy (\\%) & Macro F1 (\\%) & Worst client (\\%) & ECE (\\%) \\\\", "\\midrule"]
    for method in METHODS:
        cells = [method]
        for col, stdcol, _ in metrics:
            value = 100 * u.loc[method, col]
            text = f"{value:.2f}"
            if stdcol: text += f"$\\pm${100*u.loc[method,stdcol]:.2f}"
            if ranks[method][col] == 1: text = f"\\textbf{{{text}}}"
            elif ranks[method][col] == 2: text = f"\\underline{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (RESULTS / "tables" / "main_urbansound_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # ESC-50 uses every official fold as test once, with three fixed seeds per fold.
    esc = summary[summary.dataset == "esc50"].set_index("method")
    if set(METHODS).issubset(esc.index):
        esc = esc.loc[METHODS]
        ranks = {m: {} for m in METHODS}
        for col, _, high in metrics:
            ordered = esc[col].sort_values(ascending=not high).index.tolist()
            for rank, method in enumerate(ordered, 1):
                ranks[method][col] = rank
        lines = ["\\begin{tabular}{lcccc}", "\\toprule",
                 "Method & Accuracy (\\%) & Macro F1 (\\%) & Worst client (\\%) & ECE (\\%) \\\\", "\\midrule"]
        for method in METHODS:
            cells = [method]
            for col, stdcol, _ in metrics:
                value = 100 * esc.loc[method, col]
                shown = f"{value:.2f}"
                if stdcol:
                    shown += f"$\\pm${100 * esc.loc[method, stdcol]:.2f}"
                if ranks[method][col] == 1:
                    shown = f"\\textbf{{{shown}}}"
                elif ranks[method][col] == 2:
                    shown = f"\\underline{{{shown}}}"
                cells.append(shown)
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (RESULTS / "tables" / "main_esc50_table.tex").write_text("\n".join(lines), encoding="utf-8")

        esc_seedwise = pd.read_csv(RESULTS / "tables" / "main_comparison_seedwise.csv")
        esc_seedwise = esc_seedwise[esc_seedwise.dataset == "esc50"]
        ours = esc_seedwise[esc_seedwise.method == "FedSPECTRA"].set_index(["fold", "seed"])
        esc_rows = []
        for method in METHODS[:-1]:
            base = esc_seedwise[esc_seedwise.method == method].set_index(["fold", "seed"])
            common = ours.index.intersection(base.index)
            for metric_name, column in [("accuracy", "pooled_accuracy"), ("macro_f1", "pooled_macro_f1")]:
                delta = 100 * (ours.loc[common, column] - base.loc[common, column]).to_numpy()
                samples = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(1)
                esc_rows.append({"baseline": method, "metric": metric_name, "paired_runs": len(delta),
                                 "gain_pp": delta.mean(), "ci_low_pp": np.quantile(samples, .025),
                                 "ci_high_pp": np.quantile(samples, .975),
                                 "paired_t_p": ttest_1samp(delta, 0.0).pvalue,
                                 "wins": int((delta > 0).sum())})
        pd.DataFrame(esc_rows).to_csv(RESULTS / "tables" / "esc50_significance_tests.csv", index=False)

    ab = pd.read_csv(RESULTS / "tables" / "ablation_urbansound8k.csv")
    ab.to_csv(RESULTS / "tables" / "ablation_urbansound8k.csv", index=False)
    if "macro_f1_mean" in ab.columns:
        metric_cols = ["accuracy_mean", "macro_f1_mean", "ece_mean"]
    else:
        metric_cols = ["pooled_accuracy", "pooled_macro_f1", "pooled_ece"]
    ranks = {
        metric_cols[0]: ab[metric_cols[0]].rank(ascending=False, method="min"),
        metric_cols[1]: ab[metric_cols[1]].rank(ascending=False, method="min"),
        metric_cols[2]: ab[metric_cols[2]].rank(ascending=True, method="min"),
    }
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "Variant & Accuracy (\\%) & Macro F1 (\\%) & ECE (\\%) \\\\", "\\midrule"]
    names = {"no_transport": "w/o spectral transport", "no_feature_prototype": "w/o feature prototype",
             "no_label_reliability": "w/o label reliability", "no_spectral_reliability": "w/o spectral reliability",
             "no_trajectory": "w/o trajectory stabilization", "full": "FedSPECTRA"}
    for i, row in ab.iterrows():
        vals = [100 * row[c] for c in metric_cols]
        cell = []
        for c, v in zip(metric_cols, vals):
            text = f"{v:.2f}"
            if ranks[c].loc[i] == 1: text = f"\\textbf{{{text}}}"
            elif ranks[c].loc[i] == 2: text = f"\\underline{{{text}}}"
            cell.append(text)
        lines.append(names.get(row.variant, row.variant) + " & " + " & ".join(cell) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (RESULTS / "tables" / "ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")

    staged = pd.read_csv(RESULTS / "tables" / "staged_ablation.csv")
    piv_f1 = staged.pivot(index="stage", columns="shift", values="macro_f1_mean") * 100
    piv_delta = staged.pivot(index="stage", columns="shift", values="delta_macro_f1_pp")
    order = ["trajectory", "feature", "transport", "label_reliability", "full"]
    names = {"trajectory": "Trajectory stabilizer", "feature": "+ feature prototype",
             "transport": "+ spectral transport", "label_reliability": "+ label reliability",
             "full": "+ spectral reliability (full)"}
    ranks = {shift: piv_f1[shift].rank(ascending=False, method="min") for shift in ["standard", "strong"]}
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Stage & Std. F1 (\\%) & $\\Delta$ (pp) & Strong F1 (\\%) & $\\Delta$ (pp) \\\\", "\\midrule"]
    for stage in order:
        cells = []
        for shift in ["standard", "strong"]:
            value = piv_f1.loc[stage, shift]
            shown = f"{value:.2f}"
            if ranks[shift].loc[stage] == 1:
                shown = f"\\textbf{{{shown}}}"
            elif ranks[shift].loc[stage] == 2:
                shown = f"\\underline{{{shown}}}"
            delta = piv_delta.loc[stage, shift]
            delta_shown = "--" if pd.isna(delta) else f"{delta:+.2f}"
            cells.extend([shown, delta_shown])
        lines.append(names[stage] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (RESULTS / "tables" / "staged_ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    setup_style()
    main_performance()
    mechanism_figure()
    calibration_fairness_figure()
    statistical_tables()
    paper_figures = RESULTS.parent.parent / "paper" / "ICASSP" / "figures"
    paper_figures.mkdir(parents=True, exist_ok=True)
    for name in ("main_performance.pdf", "robustness_mechanism.pdf", "calibration_fairness.pdf"):
        source = RESULTS / "figures" / name
        (paper_figures / name).write_bytes(source.read_bytes())
    print("analysis, tables, and synchronized PDF figures completed")


if __name__ == "__main__":
    main()
