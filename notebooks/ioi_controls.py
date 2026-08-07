"""Controlled patching experiments on IOI, 0-shot only.

Imports ioi_intervention for its whole apparatus (SGLang AV, target model, AR
critic, prompt builder, patch hook). That import launches/attaches SGLang and
loads two 27B models -- deliberate here, unlike in plot_ioi.py. The sweep in
ioi_intervention does NOT run, since it sits under `if __name__ == "__main__"`.

Two passes, so the expensive part happens once per sample:

  pass 1 (once per sample): baseline generate -> capture h_orig -> av() ->
          explanation -> ar() -> h_hat.  One 384-token AV generation each;
          this dominates the runtime.
  pass 2 (cheap): every condition is a rescale plus one 5-token generate,
          reusing the vectors from pass 1. The mismatched control costs no
          extra AV or AR call at all -- it is another sample's h_hat.

Conditions:
  baseline    no patch at all
  nla         patch with h_hat (the standard round-trip)
  mismatch    patch with h_hat from sample i+1 (cyclic) -- same distribution,
              same length, only the content differs
  random      random direction scaled to ||h_orig|| -- direction destroyed,
              magnitude correct: the damage floor
  alpha=A     (h_hat/||h_hat||) * A * ||h_orig|| -- direction held, magnitude
              swept. A=0 is ablation, A=1 is norm-matched, A~1.5 is what the
              nla condition already does.

0-shot only on purpose: the existing sweep saturates at >=2 shots, so the
other shot counts would just print 1.00 repeatedly.

    python ioi_controls.py [n_samples]   # default 50; 3 for a smoke test; 200 = all
    python ioi_controls.py --replay      # rerun pass 2 on the SAVED explanations

--replay exists because SGLang's greedy decoding is not bit-stable run to run:
two identical full runs moved `nla` 0.860->0.840 and `mismatch` 0.800->0.820
while every deterministic condition (baseline, random, alpha=0) reproduced
exactly. Replaying pins the explanations so patch conditions are repeatable.
"""
from __future__ import annotations

import json
import random
import sys

import numpy as np
import pandas as pd
import torch

import ioi_intervention as ii

_pos = [a for a in sys.argv[1:] if not a.startswith("-")]
REPLAY = "--replay" in sys.argv
N_SAMPLES = int(_pos[0]) if _pos else 50
N_SHOTS = 0
ALPHAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0]

NPZ_PATH = ii.OUT_DIR / "ioi_controls_0shot_vectors.npz"
JSON_PATH = ii.OUT_DIR / "ioi_controls_0shot_explanations.json"

print(f"\n[controls] n_samples={N_SAMPLES}  n_shots={N_SHOTS}  alphas={ALPHAS}"
      f"  replay={REPLAY}")

pairs = ii.load_json_pairs(ii.IOI_DATA)

# Same query set the sweep uses: run_config re-seeds its own generator with
# SEED on every call, so these are the identical examples, directly comparable.
# The original 50 are drawn first and kept as a prefix, so a larger run is a
# strict superset of the smaller one -- random.sample with the same seed but a
# different k does NOT return a prefix of the bigger draw.
q_rng = random.Random(ii.SEED)
n = min(N_SAMPLES, len(pairs))
_first = q_rng.sample(range(len(pairs)), min(50, len(pairs)))
_rest = [i for i in range(len(pairs)) if i not in set(_first)]
random.Random(ii.SEED + 1).shuffle(_rest)
query_idxs = (_first + _rest)[:n]
assert len(query_idxs) == n, f"{len(query_idxs)=} != {n=}"
demo_rng = random.Random(ii.SEED)


# --------------------------------------------------------------------------
# Pass 1 -- one AV + one AR call per sample
# --------------------------------------------------------------------------
samples = []

if REPLAY:
    # Reuse the stored explanations/vectors instead of regenerating them.
    # SGLang's greedy decoding is not bit-stable across runs (batch composition
    # changes reduction order), so a fresh AV pass shifts a few explanations and
    # moves every patched condition by ~1 sample. Replaying pins them.
    print(f"\n{'=' * 78}\nPASS 1 (REPLAY): loading stored vectors\n{'=' * 78}")
    _d = np.load(NPZ_PATH)
    _meta = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert len(_meta) == len(_d["h_orig"]), "npz/json length mismatch"
    for k, rec in enumerate(_meta):
        # Baseline is a greedy unpatched generate -- deterministic, and cheap
        # enough to recompute rather than store.
        text_baseline, _ = ii.run_with_patch(
            rec["prompt"], patch_vec=None, max_new_tokens=ii.MAX_NEW_TOKENS
        )
        pred_baseline = ii.extract_word_answer(text_baseline)
        samples.append({
            "sample_idx": k,
            "gold": rec["gold"],
            "prompt": rec["prompt"],
            "explanation": rec["explanation"],
            "h_orig": torch.from_numpy(_d["h_orig"][k]).float(),
            "h_hat": torch.from_numpy(_d["h_hat"][k]).float(),
            "pred_baseline": pred_baseline,
            "correct_baseline": ii.is_correct(pred_baseline, rec["gold"],
                                              case_sensitive=False),
        })
    n = len(samples)
    print(f"[replay] {n} samples from {NPZ_PATH.name}")

if not REPLAY:
    print(f"\n{'=' * 78}\nPASS 1: baseline, AV, AR  ({n} samples)\n{'=' * 78}")

for i, qi in enumerate(query_idxs if not REPLAY else []):
    prompt, gold = ii.build_icl_prompt(pairs, qi, N_SHOTS, demo_rng)
    text_baseline, h_orig = ii.run_with_patch(
        prompt, patch_vec=None, max_new_tokens=ii.MAX_NEW_TOKENS
    )
    pred_baseline = ii.extract_word_answer(text_baseline)
    correct_baseline = ii.is_correct(pred_baseline, gold, case_sensitive=False)

    explanation = ii.av(h_orig)
    h_hat = ii.ar(explanation).squeeze()

    samples.append({
        "sample_idx": i,
        "gold": gold,
        "prompt": prompt,
        "h_orig": h_orig,
        "h_hat": h_hat,
        "explanation": explanation,
        "pred_baseline": pred_baseline,
        "correct_baseline": correct_baseline,
    })
    m = ii.recon_metrics(h_orig, h_hat)
    print(f"  [{i:>3}/{n}] gold={gold!r:<12} baseline={pred_baseline!r:<12} "
          f"correct={correct_baseline!s:<5} "
          f"||h||={m['norm_orig']:>9.1f} ||h_hat||={m['norm_hat']:>9.1f} "
          f"ratio={m['norm_hat'] / m['norm_orig']:.3f} cos={m['cos']:.4f}")

assert len(samples) >= 2, "need >=2 samples for the mismatched control"

# Dump vectors and texts before pass 2 runs, so any later similarity analysis
# is offline forever and a pass-2 crash still leaves the expensive part on disk.
# Skipped under --replay: the dump is the input there, don't rewrite it.
ii.OUT_DIR.mkdir(parents=True, exist_ok=True)
if not REPLAY:
    np.savez_compressed(
        NPZ_PATH,
        h_orig=np.stack([s["h_orig"].detach().float().cpu().numpy() for s in samples]),
        h_hat=np.stack([s["h_hat"].detach().float().cpu().numpy() for s in samples]),
        sample_idx=np.array([s["sample_idx"] for s in samples]),
    )
    JSON_PATH.write_text(
        json.dumps([{k: s[k] for k in ("sample_idx", "gold", "prompt", "explanation")}
                    for s in samples], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[ok] vectors      -> {NPZ_PATH}")
    print(f"[ok] explanations -> {JSON_PATH}")


# --------------------------------------------------------------------------
# Pass 2 -- cheap: rescale + one 5-token generate per condition
# --------------------------------------------------------------------------
print(f"\n{'=' * 78}\nPASS 2: patch conditions\n{'=' * 78}")

rows = []


def record(s, condition, patch_vec, alpha=None, verbose=True):
    """Run one patched generation and store the row."""
    if patch_vec is None:
        pred = s["pred_baseline"]
        correct = s["correct_baseline"]
        m = {"norm_orig": s["h_orig"].float().norm().item(),
             "norm_hat": float("nan"), "cos": float("nan"),
             "mse_nrm": float("nan")}
    else:
        text, _ = ii.run_with_patch(
            s["prompt"], patch_vec=patch_vec, max_new_tokens=ii.MAX_NEW_TOKENS
        )
        pred = ii.extract_word_answer(text)
        correct = ii.is_correct(pred, s["gold"], case_sensitive=False)
        # Uniform schema: metrics of the PATCH VECTOR against h_orig, whatever
        # the patch happens to be. For `nla` this is the reconstruction error;
        # for `random` it is the floor; for alpha rows it tracks the rescale.
        m = ii.recon_metrics(s["h_orig"], patch_vec)

    row = {
        "sample_idx": s["sample_idx"], "gold": s["gold"],
        "condition": condition, "alpha": alpha,
        "pred": pred, "correct": bool(correct),
        "norm_orig": m["norm_orig"], "norm_patch": m["norm_hat"],
        "cos_patch_h": m["cos"], "mse_nrm_patch_h": m["mse_nrm"],
    }
    rows.append(row)
    if verbose:
        a = "" if alpha is None else f" a={alpha:<5}"
        print(f"  [{s['sample_idx']:>3}] {condition:<10}{a} pred={pred!r:<12} "
              f"correct={str(correct):<5} ||patch||={row['norm_patch']:>9.1f} "
              f"cos={row['cos_patch_h']:>7.4f}")
    return row


for s in samples:
    h_orig = s["h_orig"]
    h_hat = s["h_hat"]
    n_orig = h_orig.float().norm().item()
    unit_hat = (h_hat.float() / h_hat.float().norm().clamp_min(1e-12)).cpu()

    record(s, "baseline", None)
    record(s, "nla", h_hat)

    # Mismatched: another sample's h_hat. No extra AV/AR call.
    other = samples[(s["sample_idx"] + 1) % len(samples)]
    record(s, "mismatch", other["h_hat"])

    # Random direction at the correct magnitude. Keyed per sample so the run
    # is reproducible regardless of ordering.
    g = torch.Generator().manual_seed(ii.SEED * 100003 + s["sample_idx"])
    rv = torch.randn(unit_hat.shape[-1], generator=g)
    record(s, "random", rv / rv.norm() * n_orig)

    # Magnitude sweep, direction held at h_hat's.
    for a in ALPHAS:
        record(s, "alpha", unit_hat * (a * n_orig), alpha=a)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
df = pd.DataFrame(rows)
# Replay writes its own file so it never clobbers the generating run's CSV.
csv_path = ii.OUT_DIR / ("ioi_controls_0shot_replay.csv" if REPLAY
                         else "ioi_controls_0shot.csv")
ii.OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_path, index=False)
print(f"\n[ok] {len(df)} rows -> {csv_path}")

print(f"\n{'=' * 78}\nACCURACY BY CONDITION  (n={n}, 0-shot)\n{'=' * 78}")
named = df[df["condition"] != "alpha"]
for cond, sub in named.groupby("condition"):
    print(f"  {cond:<10} acc={sub['correct'].mean():.3f}  "
          f"cos(patch,h)={sub['cos_patch_h'].mean():>7.4f}  "
          f"||patch||={sub['norm_patch'].mean():>9.1f}")

print(f"\n{'-' * 78}\nMAGNITUDE SWEEP  (direction held at h_hat)\n{'-' * 78}")
print(f"  {'alpha':>6} {'acc':>7} {'||patch||':>11}")
for a, sub in df[df["condition"] == "alpha"].groupby("alpha"):
    print(f"  {a:>6} {sub['correct'].mean():>7.3f} {sub['norm_patch'].mean():>11.1f}")

# --------------------------------------------------------------------------
# Free byproduct: how different ARE the reconstructions across samples?
# If these cosines sit near 1, the mismatched control had nothing to perturb.
# --------------------------------------------------------------------------
print(f"\n{'-' * 78}\nCROSS-SAMPLE SIMILARITY\n{'-' * 78}")


def _pairwise_cos(vs):
    V = torch.stack([v.float().cpu().squeeze() for v in vs])
    V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    C = V @ V.T
    off = C[~torch.eye(len(vs), dtype=torch.bool)]
    return off.mean().item(), off.median().item(), off.min().item(), off.max().item()


for name, vs in (("h_hat_i vs h_hat_j", [s["h_hat"] for s in samples]),
                 ("h_orig_i vs h_orig_j", [s["h_orig"] for s in samples])):
    mean, med, lo, hi = _pairwise_cos(vs)
    print(f"  {name:<22} mean={mean:.4f} median={med:.4f} min={lo:.4f} max={hi:.4f}")

own = df[df["condition"] == "nla"]["cos_patch_h"].mean()
mis = df[df["condition"] == "mismatch"]["cos_patch_h"].mean()
print(f"  cos(h_hat_i, h_orig_i) [own]      = {own:.4f}")
print(f"  cos(h_hat_j, h_orig_i) [mismatch] = {mis:.4f}")
print(f"  -> mismatch costs {own - mis:.4f} in cosine")
print("\ndone")
