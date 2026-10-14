"""Aggregate per-seed revision runs into the tables and figures reported in the paper.

Reads outputs/revision/<variant>_s<seed>.json and writes:
  summary.json             mean/SD/95% CI per variant and metric, communication, latency,
                           per-family acceptance, AF-MFA policy statistics, paired tests
  fig_convergence.png      validation ROC-AUC vs round and vs cumulative traffic
  fig_attack_policy.png    per-family acceptance rate and AF-MFA risk-band distribution
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy import stats
from config import load_config

METRICS = ["accuracy", "precision", "recall", "f1", "far", "frr", "roc_auc", "pr_auc", "pr_auc_attack"]
LAT = ["t_fdl", "t_hce", "t_afmfa", "t_hash", "t_store", "t_lraft", "p95_total_ms"]
NAMES = {"fdl": "FDL (all clients)", "fdl_cs": "FDL + odd-even scheduling",
         "fdl_cs_hce": "FDL + scheduling + HCE", "ctx_only": "Full, context-risk attributes only"}
COL = {"fdl": "#2a78d6", "fdl_cs": "#eb6834", "fdl_cs_hce": "#1baf7a"}
BANDS = ["low", "medium", "high", "critical"]
BAND_COL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def msd(v):
    v = np.asarray(v, float); n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": float(v.mean()), "sd": sd, "ci_low": float(v.mean() - h), "ci_high": float(v.mean() + h), "n": n, "values": v.tolist()}


def load(out):
    runs = {}
    for f in sorted(glob.glob(str(Path(out) / "*_s*.json"))):
        r = json.load(open(f)); runs.setdefault(r["variant"], []).append(r)
    for v in runs: runs[v].sort(key=lambda r: r["seed"])
    return runs


def summarize(runs):
    S = {}
    for v, rs in runs.items():
        d = {m: msd([r["metrics"][m] for r in rs]) for m in METRICS}
        d["seeds"] = [r["seed"] for r in rs]
        d["comm_total_mb"] = msd([(r["communication"]["upload_bytes"] + r["communication"]["download_bytes"]) / 1e6 for r in rs])
        d["comm_upload_mb"] = msd([r["communication"]["upload_bytes"] / 1e6 for r in rs])
        d["prototype_upload_kb"] = msd([r["communication"]["prototype_upload_bytes"] / 1e3 for r in rs])
        d["model_params"] = rs[0]["communication"]["model_params"]; d["hce_params"] = rs[0]["communication"]["hce_params"]
        d["latency_ms"] = {k: msd([r["latency_ms"][k] for r in rs]) for k in LAT if k in rs[0]["latency_ms"]}
        d["train_minutes"] = msd([r["train_seconds"] / 60 for r in rs])
        d["encoded_dim"] = sorted({r["data"]["encoded_dim"] for r in rs})
        d["client_sizes"] = [r["data"]["client_sizes"] for r in rs]
        d["client_attack_share"] = [r["data"]["client_attack_share"] for r in rs]
        types = sorted(rs[0]["per_attack_type"])
        d["per_attack_type"] = {t: {"n": int(np.mean([r["per_attack_type"][t]["n"] for r in rs])),
                                    "accepted": msd([r["per_attack_type"][t]["accepted_rate"] for r in rs])} for t in types}
        pol = {}
        for grp in ("legitimate", "malicious"):
            pol[grp] = {k: msd([r["af_mfa_policy"][grp][k] for r in rs]) for k in ("mean_factors", "mean_assurance", "mean_factor_latency_ms", "step_up_rate")}
            pol[grp]["band_share"] = {b: msd([r["af_mfa_policy"][grp]["band_share"][b] for r in rs]) for b in BANDS}
        pol["decisions"] = rs[0]["af_mfa_policy"]["decisions"]; pol["fixed"] = rs[0]["af_mfa_policy"]["fixed_password_otp"]
        d["af_mfa"] = pol
        hist = np.array([[h["val_roc_auc"] for h in r["history"]] for r in rs])
        cum = np.array([[h["cum_bytes"] / 1e6 for h in r["history"]] for r in rs])
        d["curve"] = {"auc_mean": hist.mean(0).tolist(), "auc_sd": hist.std(0, ddof=1).tolist() if len(rs) > 1 else [0] * hist.shape[1],
                      "cum_mb": cum.mean(0).tolist()}
        final = hist[:, -1]
        d["rounds_to_99pct_final_auc"] = msd([int(np.argmax(h >= 0.99 * f)) + 1 for h, f in zip(hist, final)])
        S[v] = d
    # paired tests on common seeds
    tests = []
    pairs = [("fdl", "fdl_cs"), ("fdl_cs", "fdl_cs_hce"), ("fdl", "fdl_cs_hce"), ("fdl_cs_hce", "ctx_only")]
    for a, b in pairs:
        if a in runs and b in runs:
            ca = {r["seed"]: r for r in runs[a]}; cb = {r["seed"]: r for r in runs[b]}
            common = sorted(set(ca) & set(cb))
            if len(common) >= 3:
                for m in ("accuracy", "f1", "roc_auc", "far"):
                    x = np.array([ca[s]["metrics"][m] for s in common]); y = np.array([cb[s]["metrics"][m] for s in common])
                    t, p = stats.ttest_rel(y, x); dd = (y - x)
                    tests.append({"a": a, "b": b, "metric": m, "n": len(common), "mean_diff": float(dd.mean()),
                                  "t": float(t), "p": float(p), "cohen_dz": float(dd.mean() / (dd.std(ddof=1) + 1e-12))})
    S["paired_tests"] = tests
    return S


def figures(S, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.linewidth": 0.6, "savefig.dpi": 600})
    variants = [v for v in ("fdl", "fdl_cs", "fdl_cs_hce") if v in S]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    markers = {"fdl": "o", "fdl_cs": "s", "fdl_cs_hce": "^"}
    for v in variants:
        c = S[v]["curve"]; m = np.array(c["auc_mean"]); sd = np.array(c["auc_sd"]); x = np.arange(1, len(m) + 1)
        for ax, xx in ((axes[0], x), (axes[1], np.array(c["cum_mb"]))):
            ax.plot(xx, m, color=COL[v], lw=1.4, marker=markers[v], markevery=7, ms=3.5, label=NAMES[v])
            ax.fill_between(xx, m - sd, m + sd, color=COL[v], alpha=0.15, lw=0)
    for ax, lab, t in ((axes[0], "Communication round", "(a)"), (axes[1], "Cumulative traffic (MB)", "(b)")):
        ax.set_xlabel(lab); ax.set_ylabel("Validation ROC-AUC"); ax.set_ylim(0.985, 1.0005)
        ax.grid(True, color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_title(t, loc="left", fontsize=8)
    axes[0].set_xlim(0, len(S[variants[0]]["curve"]["auc_mean"]) * 1.03)
    axes[1].set_xlim(0, max(S[v]["curve"]["cum_mb"][-1] for v in variants) * 1.03)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(Path(out) / "fig_convergence.png"); plt.close(fig)

    full = S["fdl_cs_hce"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
    pa = full["per_attack_type"]; fams = [t for t in pa if t != "normal"]
    fams.sort(key=lambda t: -pa[t]["accepted"]["mean"])
    vals = np.array([pa[t]["accepted"]["mean"] * 100 for t in fams]); err = np.array([pa[t]["accepted"]["sd"] * 100 for t in fams])
    ax = axes[0]; y = np.arange(len(fams))
    ax.barh(y, vals, xerr=err, color="#2a78d6", height=0.6, error_kw={"lw": 0.6, "capsize": 1.5, "ecolor": "#52514e"})
    ax.set_yticks(y); ax.set_yticklabels([{"mitm": "MITM", "xss": "XSS", "dos": "DoS", "ddos": "DDoS"}.get(t, t.capitalize()) for t in fams])
    ax.invert_yaxis(); ax.set_xlabel("Accepted as legitimate (%)")
    for yi, v_, e in zip(y, vals, err): ax.text(v_ + e + 0.05 * max(1e-9, vals.max() + err.max()), yi, f"{v_:.2f}", va="center", fontsize=6.5, color="#0b0b0b")
    ax.set_xlim(0, (vals.max() + err.max()) * 1.35 + 1e-6)
    ax.set_title("(a)", loc="left", fontsize=8)
    ax = axes[1]
    for gi, grp in enumerate(("legitimate", "malicious")):
        left = 0
        for b, col in zip(BANDS, BAND_COL):
            w = full["af_mfa"][grp]["band_share"][b]["mean"] * 100
            ax.barh(gi, w, left=left, color=col, height=0.55, edgecolor="#fcfcfb", lw=1)
            if w > 9: ax.text(left + w / 2, gi, f"{w:.1f}", ha="center", va="center", fontsize=6.5, color="#0b0b0b" if b == "low" else "#fcfcfb")
            left += w
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legitimate", "Malicious"]); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("Test sessions per AF-MFA risk band (%)")
    from matplotlib.patches import Patch
    ax.legend([Patch(color=c) for c in BAND_COL], [b.capitalize() for b in BANDS], ncol=4, frameon=False, fontsize=6.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.32), handlelength=1, columnspacing=0.8)
    ax.set_title("(b)", loc="left", fontsize=8)
    for a_ in axes:
        for s in ("top", "right"): a_.spines[s].set_visible(False)
        a_.grid(True, axis="x", color="#e4e3df", lw=0.5); a_.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(Path(out) / "fig_attack_policy.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
    cfg = load_config(a.config); out = Path(cfg["experiment"]["output_dir"]) / "revision"
    S = summarize(load(out)); json.dump(S, open(out / "summary.json", "w"), indent=1)
    figures(S, out); print(out / "summary.json")
