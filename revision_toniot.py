"""Primary ToN-IoT protocol used for the reported BCFDL-HCE results.

One protocol produces every headline number:
  * ToN-IoT Train_Test_Network (211,043 records), endpoint identifiers excluded;
  * stratified 64/16/20 train/validation/test split per seed, preprocessing fitted on train only;
  * 10 Dirichlet(0.5) clients, FedAvg, 100 rounds, 5 local epochs, Adam 1e-3, batch 64;
  * clipping (norm 1.0) + weight-correct pairwise-mask secure aggregation + robust screening;
  * the single global model after round 100 is scored on the pooled held-out test split;
  * the acceptance threshold is chosen on the validation split (min FAR + FRR) and frozen;
  * five seeds (42-46); tables report mean +/- SD over seeds.

Legitimate activity is the positive class in every reported metric (y_legit = 1 - label).

Usage:
  python experiments/revision_toniot.py --config config.yaml --variant fdl --seeds 42 43
  python experiments/summarize_revision.py --config config.yaml
"""
from __future__ import annotations
import argparse, io, json, os, sys, tempfile, time, uuid
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from config import load_config
from data import load_toniot_csv, make_client_loaders
from models import AuthenticationNet
from hce import HashContextEncoder
from federated import FederatedTrainer
from af_mfa import AFMFASelector, DEFAULT_FACTORS, combined_assurance, assurance_required
from ledger import PermissionedLedger, LocalCAS, AuthenticationTransaction, LimitedRaftCoordinator
from utils import ensure_dir, save_json, seed_everything, sha256_bytes, environment_snapshot

# ---------------------------------------------------------------- feature mapping (Table in paper)
IDENTIFIERS = ["src_ip", "dst_ip", "src_port"]
FEATURE_GROUPS = {
    "session_behaviour": ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "dst_pkts",
                          "src_ip_bytes", "dst_ip_bytes", "conn_state"],
    "service_protocol": ["proto", "service", "dst_port"],
    "credential_handshake": ["ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "ssl_subject",
                             "ssl_issuer", "http_user_agent", "http_method", "http_version",
                             "http_status_code", "http_trans_depth"],
    "name_resolution": ["dns_query", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA",
                        "dns_rejected"],
    "payload_anomaly": ["http_uri", "http_request_body_len", "http_response_body_len", "http_orig_mime_types",
                        "http_resp_mime_types", "weird_name", "weird_addl", "weird_notice"],
}
CONTEXT_RISK = FEATURE_GROUPS["session_behaviour"] + FEATURE_GROUPS["service_protocol"] + FEATURE_GROUPS["credential_handshake"]

# variant -> (odd-even cluster scheduling, HCE, predictor restriction)
VARIANTS = {
    "fdl":         dict(cluster=False, hce=False, keep=None),
    "fdl_cs":      dict(cluster=True,  hce=False, keep=None),
    "fdl_cs_hce":  dict(cluster=True,  hce=True,  keep=None),   # + AF-MFA + ledger = full BCFDL-HCE
    "ctx_only":    dict(cluster=True,  hce=True,  keep=CONTEXT_RISK),
}


def scores(model, loader, device):
    model.eval(); out = []; ys = []
    with torch.no_grad():
        for x, y in loader:
            out.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu()); ys.append(y)
    p_attack = torch.cat(out).numpy().astype(np.float64)
    return p_attack, torch.cat(ys).numpy()


def select_threshold(s_legit, y_legit, grid=2001):
    """Acceptance threshold on the validation split: accept iff s >= tau, tau = argmin FAR + FRR."""
    cands = np.unique(np.quantile(s_legit, np.linspace(0, 1, grid)))
    mal = y_legit == 0; leg = y_legit == 1
    best = (np.inf, 0.5)
    for t in cands:
        acc = s_legit >= t
        far = acc[mal].mean() if mal.any() else 0.0
        frr = (~acc[leg]).mean() if leg.any() else 0.0
        if far + frr < best[0]: best = (far + frr, float(t))
    return best[1]


def legit_metrics(s_legit, y_legit, tau):
    pred = (s_legit >= tau).astype(int)
    tp = int(((pred == 1) & (y_legit == 1)).sum()); tn = int(((pred == 0) & (y_legit == 0)).sum())
    fp = int(((pred == 1) & (y_legit == 0)).sum()); fn = int(((pred == 0) & (y_legit == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"accuracy": (tp + tn) / len(y_legit), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(1e-12, prec + rec),
            "far": fp / max(1, fp + tn), "frr": fn / max(1, fn + tp),
            "roc_auc": float(roc_auc_score(y_legit, s_legit)),
            "pr_auc": float(average_precision_score(y_legit, s_legit)),
            "pr_auc_attack": float(average_precision_score(1 - y_legit, 1 - s_legit)),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": tau}


# ---------------------------------------------------------------- AF-MFA policy behaviour
def risk_band(r, cfg):
    if r < cfg["low_risk_threshold"]: return "low"
    if r < cfg["medium_risk_threshold"]: return "medium"
    if r < cfg["high_risk_threshold"]: return "high"
    return "critical"


def af_mfa_policy(risk, y_legit, cfg, seed):
    """Band-level GA decisions applied to every test session; fixed-factor MFA as reference."""
    band_risk = {"low": 0.15, "medium": 0.50, "high": 0.75, "critical": 0.95}
    sel = AFMFASelector(cfg, seed=seed)
    decisions = {b: sel.select(r) for b, r in band_risk.items()}
    bands = np.array([risk_band(r, cfg) for r in risk])
    out = {"decisions": decisions}
    for name, mask in [("legitimate", y_legit == 1), ("malicious", y_legit == 0)]:
        b = bands[mask]
        out[name] = {
            "n": int(mask.sum()),
            "band_share": {k: float((b == k).mean()) for k in band_risk},
            "mean_factors": float(np.mean([len(decisions[k]["factors"]) for k in b])),
            "mean_assurance": float(np.mean([decisions[k]["achieved_assurance"] for k in b])),
            "mean_factor_latency_ms": float(np.mean([decisions[k]["estimated_latency_ms"] for k in b])),
            "step_up_rate": float((b != "low").mean()),
        }
    fixed = [f for f in DEFAULT_FACTORS if f.name in ("password", "otp")]
    out["fixed_password_otp"] = {"factors": [f.name for f in fixed], "assurance": combined_assurance(fixed),
                                 "latency_ms": sum(f.latency_ms for f in fixed)}
    return out


# ---------------------------------------------------------------- per-session latency
def measure_latency(model, hce, Xs, cfg, seed, workdir, with_chain):
    model.eval(); sel = AFMFASelector(cfg["af_mfa"], seed=seed)
    ledger = PermissionedLedger(Path(workdir) / "ledger", int(cfg["blockchain"]["validator_nodes"]))
    cas = LocalCAS(Path(workdir) / "cas")
    decisions = {}
    rows = []
    with torch.no_grad():
        for i in range(len(Xs)):
            x = Xs[i:i + 1]; t0 = time.perf_counter()
            logits, z = model(x, return_embedding=True)
            risk = float(torch.softmax(logits, 1)[0, 1]); t1 = time.perf_counter()
            code = b""
            if hce is not None:
                h = hce(z); bits = (h[0] >= 0).numpy().astype(np.uint8); code = np.packbits(bits).tobytes()
            t2 = time.perf_counter()
            band = risk_band(risk, cfg["af_mfa"])
            d = sel.select(risk)  # full GA search per session (no caching) to measure worst-case policy cost
            decisions[band] = d["factors"]; t3 = time.perf_counter()
            row = {"t_fdl": t1 - t0, "t_hce": t2 - t1, "t_afmfa": t3 - t2, "t_hash": 0.0, "t_lraft": 0.0, "t_store": 0.0}
            if with_chain:
                event = json.dumps({"session": str(uuid.uuid4()), "risk": risk, "band": band,
                                    "factors": d["factors"], "hce": code.hex(), "ts": time.time()}).encode()
                t4 = time.perf_counter(); dig = sha256_bytes(event); t5 = time.perf_counter()
                cid = cas.add_bytes(event); t6 = time.perf_counter()
                ledger.append(AuthenticationTransaction(str(uuid.uuid4()), "auth-service", "authentication_event",
                                                        dig, cid, time.time(), {"band": band})); t7 = time.perf_counter()
                row.update({"t_hash": t5 - t4, "t_store": t6 - t5, "t_lraft": t7 - t6})
            rows.append(row)
    res = {k: float(np.mean([r[k] for r in rows]) * 1000) for k in rows[0]}
    res["p95_total_ms"] = float(np.percentile([sum(r.values()) * 1000 for r in rows], 95))
    if with_chain:
        res["ledger_valid"] = bool(ledger.verify()); res["ledger_blocks"] = len(ledger.chain)
    return res


def lraft_failure_test(validators=5, commits=40):
    """Crash-fault behaviour of the Limited-Raft coordinator: leader crash, re-election, quorum loss."""
    log = []
    with tempfile.TemporaryDirectory() as td:
        led = PermissionedLedger(td, validators); raft = led.raft
        def commit_batch(tag):
            ts = []
            for i in range(commits):
                t = time.perf_counter()
                led.append(AuthenticationTransaction(str(uuid.uuid4()), "c", tag, "0" * 64, "cid", time.time(), {}))
                ts.append((time.perf_counter() - t) * 1000)
            return float(np.mean(ts))
        log.append({"state": "all alive", "alive": len(raft.alive), "leader": raft.leader, "term": raft.term, "commit_ms": commit_batch("a")})
        for k in range(validators):
            leader = raft.leader
            try:
                raft.crash(leader)
                ok = True; ms = commit_batch(f"after-crash-{k+1}")
            except RuntimeError as e:
                ok = False; ms = None
            log.append({"state": f"{k+1} crashed", "alive": len(raft.alive), "leader": raft.leader if ok else None,
                        "term": raft.term, "commit_ok": ok, "commit_ms": ms})
            if not ok: break
        valid = led.verify()
    return {"validators": validators, "quorum": validators // 2 + 1, "tolerated_crashes": (validators - 1) // 2,
            "log": log, "ledger_valid_after_test": valid}


# ---------------------------------------------------------------- one run
def run(cfg, variant, seed, out, latency_sessions):
    v = VARIANTS[variant]
    c = deepcopy(cfg); c["experiment"]["seed"] = seed
    c["federated"]["cluster_training"] = v["cluster"]; c["federated"]["odd_even_schedule"] = v["cluster"]
    seed_everything(seed); device = torch.device("cpu")
    d = c["datasets"]
    b = load_toniot_csv(d["toniot_csv"], target_column="label", test_size=float(d["test_size"]),
                        val_size_from_train=float(d["val_size_from_train"]), seed=seed,
                        exclude_columns=d.get("exclude_columns", IDENTIFIERS), keep_columns=v["keep"],
                        max_categories=d.get("max_categories"))
    clients = make_client_loaders(b.train, c, seed)
    val_loader = torch.utils.data.DataLoader(b.val, batch_size=4096); test_loader = torch.utils.data.DataLoader(b.test, batch_size=4096)
    model = AuthenticationNet(b.feature_dim)
    hce = HashContextEncoder(128, int(c["hce"]["bits"]), int(c["hce"]["projection_dim"])) if v["hce"] else None
    tr = FederatedTrainer(model, clients, c, device, hce, "fedavg")
    hist = []; t0 = time.perf_counter(); cum = 0
    for r in range(int(c["federated"]["rounds"])):
        rec = tr.round(r); cum += rec["upload_bytes"] + rec["download_bytes"]
        pv, yv = scores(tr.model, val_loader, device)
        rec.update({"cum_bytes": cum, "val_roc_auc": float(roc_auc_score(1 - yv, 1 - pv)),
                    "val_acc_05": float(((pv < 0.5).astype(int) == (1 - yv)).mean())})
        hist.append(rec)
        if r % 10 == 0 or r == int(c["federated"]["rounds"]) - 1:
            print(f"[{variant} s{seed}] round {r+1} val_auc={rec['val_roc_auc']:.5f} cum_MB={cum/1e6:.1f} t={time.perf_counter()-t0:.0f}s", flush=True)
    train_s = time.perf_counter() - t0
    pv, yv = scores(tr.model, val_loader, device); pt, yt = scores(tr.model, test_loader, device)
    tau = select_threshold(1 - pv, 1 - yv)
    m = legit_metrics(1 - pt, 1 - yt, tau)
    types = b.meta["test_attack_type"]; acc = (1 - pt) >= tau
    per_type = {t: {"n": int((types == t).sum()), "accepted_rate": float(acc[types == t].mean())} for t in np.unique(types)}
    policy = af_mfa_policy(pt, 1 - yt, c["af_mfa"], seed)
    lat_idx = np.random.default_rng(seed).choice(len(b.test), size=min(latency_sessions, len(b.test)), replace=False)
    Xs = b.test.tensors[0][torch.as_tensor(lat_idx)]
    with tempfile.TemporaryDirectory() as td:
        lat = measure_latency(tr.model, tr.hce, Xs, c, seed, td, with_chain=v["hce"])
    res = {"variant": variant, "seed": seed, "metrics": m, "per_attack_type": per_type, "af_mfa_policy": policy,
           "latency_ms": lat, "train_seconds": train_s,
           "communication": {"upload_bytes": int(sum(h["upload_bytes"] for h in hist)),
                             "download_bytes": int(sum(h["download_bytes"] for h in hist)),
                             "prototype_upload_bytes": int(sum(h["prototype_upload_bytes"] for h in hist)),
                             "model_params": int(sum(p.numel() for p in tr.model.parameters())),
                             "hce_params": int(sum(p.numel() for p in tr.hce.parameters())) if tr.hce is not None else 0},
           "history": hist, "data": {"encoded_dim": b.meta["encoded_dim"], "predictors": b.meta["predictors"],
                                     "n_train": len(b.train), "n_val": len(b.val), "n_test": len(b.test),
                                     "client_sizes": [len(l.dataset) for l in clients],
                                     "client_attack_share": [float(b.train.tensors[1][torch.as_tensor(l.dataset.indices)].float().mean()) for l in clients]},
           "environment": environment_snapshot(), "torch_threads": torch.get_num_threads()}
    save_json(Path(out) / f"{variant}_s{seed}.json", res)
    np.savez_compressed(Path(out) / f"{variant}_s{seed}_test_scores.npz", p_attack=pt, y_attack=yt, attack_type=types)
    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+"); ap.add_argument("--rounds", type=int)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--latency-sessions", type=int, default=500)
    ap.add_argument("--lraft-test", action="store_true")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    cfg = load_config(a.config)
    if a.rounds: cfg["federated"]["rounds"] = a.rounds
    out = ensure_dir(Path(cfg["experiment"]["output_dir"]) / "revision")
    if a.lraft_test: save_json(out / "lraft_failure_test.json", lraft_failure_test(int(cfg["blockchain"]["validator_nodes"])))
    base = int(cfg["experiment"]["seed"]); seeds = a.seeds or [base + i for i in range(int(cfg["experiment"]["repetitions"]))]
    for s in seeds: run(cfg, a.variant, s, out, a.latency_sessions)
