from __future__ import annotations


USER_NAME = "abhinav"
VISIBLE_GPUS = "5,7"
SGLANG_PHYSICAL_GPU = "6"
LOCAL_DEVICE = "cuda:0"

import os
from pathlib import Path

HOME = Path("/home") / USER_NAME
NLA_REPO = Path(os.environ.get("NLA_REPO", HOME / "natural_language_autoencoders"))
SGLANG_REPO = Path(os.environ.get("SGLANG_REPO", HOME / "sglang"))
VENV_PYTHON = Path(os.environ.get("NLA_VENV_PYTHON", NLA_REPO / ".venv/bin/python"))
HF_CACHE = Path(os.environ.get("HF_HOME", HOME / ".cache/huggingface"))
ROOT = Path(os.environ.get("NLA_EXPERIMENT_ROOT", HOME / "experiments/nla_experiments"))
LIBNUMA_DIR = Path(
    os.environ.get(
        "NLA_LIBNUMA_DIR", NLA_REPO / "vendor/libnuma/usr/lib/x86_64-linux-gnu"
    )
)
PYTHON_INCLUDE_DIRS = [
    Path(
        os.environ.get(
            "NLA_PYTHON_INCLUDE",
            NLA_REPO / "vendor/python3.10-dev/usr/include/python3.10",
        )
    ),
    Path(
        os.environ.get(
            "NLA_PYTHON_MULTIARCH_INCLUDE",
            NLA_REPO / "vendor/python3.10-dev/usr/include",
        )
    ),
]
ROOT.mkdir(parents=True, exist_ok=True)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
if LIBNUMA_DIR.exists():
    os.environ["LD_LIBRARY_PATH"] = (
        f"{LIBNUMA_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
_existing_cpath = os.environ.get("CPATH", "")
_include_paths = [str(path) for path in PYTHON_INCLUDE_DIRS if path.exists()]
if _include_paths:
    os.environ["CPATH"] = ":".join([*_include_paths, _existing_cpath])

import gc
import json
import math
import random
import re
import string
import subprocess
import sys
import textwrap
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(NLA_REPO))
from nla_inference import NLAClient, NLACritic
from nla.schema import compute_predict_mean_baselines, normalize_activation

TARGET_MODEL = "google/gemma-3-27b-it"
ACTOR_REPO_ID = os.environ.get("NLA_ACTOR_DIR", "kitft/nla-gemma3-27b-L41-av")
AR_REPO_ID = os.environ.get("NLA_AR_DIR", "kitft/nla-gemma3-27b-L41-ar")
SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000")
SGLANG_LOG = Path.home() / "logs" / "sglang_av_server.log"
SGLANG_LOG.parent.mkdir(parents=True, exist_ok=True)
LAYER_INDEX = 41
DEVICE = LOCAL_DEVICE
SEED = 1234



def resolve_checkpoint(path_or_repo_id: str) -> str:
    """Return a local checkpoint path with nla_meta.yaml available."""
    candidate = Path(path_or_repo_id).expanduser()
    if candidate.exists():
        assert (
            candidate / "nla_meta.yaml"
        ).exists(), f"Missing nla_meta.yaml in {candidate}"
        return str(candidate)
    local_path = Path(snapshot_download(repo_id=path_or_repo_id))
    assert (
        local_path / "nla_meta.yaml"
    ).exists(), f"Missing nla_meta.yaml in downloaded snapshot {local_path}"
    return str(local_path)


ACTOR_DIR = resolve_checkpoint(ACTOR_REPO_ID)
AR_DIR = resolve_checkpoint(AR_REPO_ID)

assert NLA_REPO.exists(), f"Missing NLA repo: {NLA_REPO}"
assert SGLANG_REPO.exists(), f"Missing patched SGLang checkout: {SGLANG_REPO}"
assert VENV_PYTHON.exists(), f"Missing NLA venv Python: {VENV_PYTHON}"

print("home:", HOME)
print("NLA repo:", NLA_REPO)
print("SGLang checkout:", SGLANG_REPO)
print("HF cache:", HF_CACHE)
print("actor checkpoint:", ACTOR_DIR)
print("AR checkpoint:", AR_DIR)
print("notebook device:", DEVICE)
print(torch.__version__, torch.cuda.is_available())


UV_PYTHON_INCLUDE = f"/home/{USER_NAME}/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/include/python3.10"
assert os.path.exists(
    os.path.join(UV_PYTHON_INCLUDE, "Python.h")
), "Python.h not found at that path"
os.environ["CPATH"] = f"{UV_PYTHON_INCLUDE}:{os.environ.get('CPATH', '')}"

REAL_LIBNUMA_DIR = f"/home/{USER_NAME}/.local/lib"
assert os.path.exists(
    os.path.join(REAL_LIBNUMA_DIR, "libnuma.so.1")
), "libnuma.so.1 not found there"
os.environ["LD_LIBRARY_PATH"] = f"{REAL_LIBNUMA_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"

print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CPATH:", os.environ.get("CPATH"))


def launch_sglang_actor() -> subprocess.Popen:
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = SGLANG_PHYSICAL_GPU
    env["SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR"] = "1"
    env["PYTHONPATH"] = f"{NLA_REPO}:{env.get('PYTHONPATH', '')}"
    if LIBNUMA_DIR.exists():
        env["LD_LIBRARY_PATH"] = f"{LIBNUMA_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    include_paths = [str(path) for path in PYTHON_INCLUDE_DIRS if path.exists()]
    if include_paths:
        env["CPATH"] = ":".join([*include_paths, env.get("CPATH", "")])

    cmd = [
        str(VENV_PYTHON),
        "-m",
        "sglang.launch_server",
        "--model-path",
        ACTOR_DIR,
        "--port",
        "30000",
        "--host",
        "127.0.0.1",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.80",
        "--context-length",
        "512",
        "--attention-backend",
        "triton",
        "--disable-cuda-graph",
        "--trust-remote-code",
        "--log-level",
        "warning",
    ]
    print("Launching SGLang on physical GPU", SGLANG_PHYSICAL_GPU)
    print(" ".join(cmd))
    print("SGLang server logs ->", SGLANG_LOG)
    log_f = open(SGLANG_LOG, "ab", buffering=0)
    return subprocess.Popen(
        cmd, env=env, cwd=str(NLA_REPO), stdout=log_f, stderr=subprocess.STDOUT
    )


def sglang_is_healthy() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen(SGLANG_URL + "/health", timeout=2).read()
        return True
    except Exception:
        return False


def wait_for_sglang(timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if sglang_is_healthy():
            print("SGLang is healthy")
            return
        try:
            import urllib.request

            urllib.request.urlopen(SGLANG_URL + "/health", timeout=2).read()
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"SGLang did not become healthy: {last_error!r}")


sglang_proc = None
if sglang_is_healthy():
    print("SGLang is already healthy at", SGLANG_URL)
else:
    sglang_proc = launch_sglang_actor()
    wait_for_sglang()


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": DEVICE},
    trust_remote_code=True,
    attn_implementation="eager",
).eval()
del model.model.vision_tower
gc.collect()
torch.cuda.empty_cache()
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

client = NLAClient(ACTOR_DIR, sglang_url=SGLANG_URL)
critic = NLACritic(AR_DIR, device="cuda:1", dtype=torch.float32)
print("Target and AV client ready")


def av(h: torch.Tensor) -> str:
    v = h.detach().float().cpu().numpy()
    return client.generate(
        v, temperature=0.0, max_new_tokens=384, extract_explanation=True
    )


def ar(text: str) -> torch.Tensor:
    return critic.reconstruct(text)  # raw, unnormalized -- matches h's scale


class LastTokenPatcher:
    def __init__(self, patch_vec=None):
        self.patch_vec = patch_vec
        self.captured = None
        self._done = False  # only patch once

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Only act on the initial multi-token prefill pass, not each
        # single-token decode step that follows.
        if hidden.shape[1] > 1 and not self._done:
            self.captured = hidden[0, -1, :].detach().clone()
            if self.patch_vec is not None:
                hidden = hidden.clone()
                hidden[0, -1, :] = self.patch_vec.to(hidden.dtype).to(hidden.device)
            self._done = True
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        return output


def get_layer_module(m, layer_idx):
    return m.model.language_model.layers[layer_idx]


def run_with_patch(prompt: str, patch_vec, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    patcher = LastTokenPatcher(patch_vec)
    handle = get_layer_module(model, LAYER_INDEX).register_forward_hook(patcher)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
    finally:
        handle.remove()
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return text, patcher.captured


MAX_NEW_TOKENS = 5
SHOT_COUNTS = [0, 2, 4, 6, 8, 10]
N_PER_CONFIG = 50

DATA_DIR = NLA_REPO / "notebooks" / "datasets"
OUT_DIR = Path("/mnt/ssd-1/soar-nla") / USER_NAME
IOI_DATA = DATA_DIR / "ioi_1.jsonl"


PARAPHRASE_PROMPT_MEDIUM = (
    "Please paraphrase the text provided below, preserving its exact meaning "
    "but expressing it in substantially different words and sentence structure "
    "than the original. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "<text>\n{text}\n</text>\n\n"
    "Include your final paraphrased text in <transformed_text> tags."
)
PARAPHRASE_PROMPT_HEAVY = (
    "Please paraphrase aggressively. Preserve meaning and uncertainty, but "
    "substantially change wording, syntax, and organization. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "Text:\n{text}\n\n"
    "Respond with only the paraphrased text, wrapped exactly like this:\n"
    "<transformed_text>your paraphrase here</transformed_text>"
)


def paraphrase(
    text: str,
    prompt_template: str = PARAPHRASE_PROMPT_MEDIUM,
    max_new_tokens: int = 512,
) -> str:
    """Rewrites `text` using `prompt_template` (PARAPHRASE_PROMPT_MEDIUM or
    PARAPHRASE_PROMPT_HEAVY).
    """
    prompt = prompt_template.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    match = re.search(r"<transformed_text>(.*?)</transformed_text>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        print(f"[paraphrase] WARNING: no <transformed_text> tags. Raw[:200]={raw[:200]!r}")
        return raw


def _extract_transformed(raw: str) -> str:
    match = re.search(r"<transformed_text>(.*?)</transformed_text>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    print(f"[paraphrase] WARNING: no <transformed_text> tags. Raw[:200]={raw[:200]!r}")
    return raw


def paraphrase_batch(
    text: str, templates: list[str], max_new_tokens: int = 512
) -> list[str]:
    """Same as calling paraphrase() once per template, but in a single batched
    generate. Measured 2.12x faster with byte-identical outputs.

    padding_side MUST be "left" here: this is a decoder-only model, and right
    padding puts pad tokens between the prompt and the first generated token,
    which corrupts the continuation. The tokenizer default is "right", so it is
    flipped for the call and restored after.
    """
    prompts = [t.format(text=text) for t in templates]
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False
            )
    finally:
        tokenizer.padding_side = prev_side

    n_in = enc["input_ids"].shape[1]
    return [
        _extract_transformed(
            tokenizer.decode(row[n_in:], skip_special_tokens=True).strip()
        )
        for row in out
    ]


def load_json_pairs(path: Path) -> list[dict]:
    """One {"input": ..., "output": ...} object per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# IOI name handling. Every mecha_ioi template introduces the pair as "X and Y"
# (verified: matches on all 200 rows, gold is always one of the two), so the
# pair pattern gets both names with no stoplist to maintain.
# --------------------------------------------------------------------------
_PAIR = re.compile(r"\b([A-Z][a-z]+) and ([A-Z][a-z]+)\b")


def names_in(row: dict) -> set[str]:
    m = _PAIR.search(row["input"])
    assert m, f"no 'X and Y' pair in: {row['input']!r}"
    return set(m.groups())


def io_and_s(row: dict) -> tuple[str, str]:
    """IO is the gold (received the object); S is the repeated subject —
    the distractor the IOI task is about suppressing."""
    io = row["output"]
    (s,) = names_in(row) - {io}
    return io, s


def build_icl_prompt(
    pairs: list[dict], query_idx: int, n_shots: int, rng: random.Random
) -> tuple[str, str]:
    """Same Q/A format as executor.py's, with one IOI-specific change: demos
    are excluded by shared NAME, not by matching input string."""
    q_names = names_in(pairs[query_idx])
    pool = [
        i
        for i in range(len(pairs))
        if i != query_idx and not (names_in(pairs[i]) & q_names)
    ]
    assert len(pool) >= n_shots, f"pool {len(pool)} < {n_shots} shots at query {query_idx}"
    shot_idxs = rng.sample(pool, min(n_shots, len(pool)))
    lines = [f"Q: {pairs[i]['input']}\nA: {pairs[i]['output']}" for i in shot_idxs]
    lines.append(f"Q: {pairs[query_idx]['input']}\nA:")
    instruction = "Answer with a single word only. No punctuation, no explanation."
    prompt = instruction + "\n\n" + "\n\n".join(lines)
    return prompt, pairs[query_idx]["output"]


def extract_word_answer(text: str) -> str:
    parts = text.strip().split()
    return parts[0].strip(string.punctuation) if parts else ""    


def is_correct(pred: str, gold: str, case_sensitive: bool) -> bool:
    if not case_sensitive:
        pred, gold = pred.lower(), gold.lower()
    return pred == gold


def is_meaningfully_paraphrased(
    original: str, paraphrased: str, max_overlap: float = 0.7
) -> bool:
    orig_words = set(original.lower().split())
    para_words = set(paraphrased.lower().split())
    if not orig_words:
        return True
    overlap = len(orig_words & para_words) / len(orig_words)
    return overlap < max_overlap

def cleanup():
    if "client" in globals() and hasattr(client, "_http"):
        client._http.close()
        print("Closed NLA client HTTP session")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if "sglang_proc" in globals() and sglang_proc is not None:
        sglang_proc.terminate()
        sglang_proc.wait(timeout=30)
        print("Stopped SGLang launched by this notebook")
    else:
        print("No SGLang subprocess launched by this file")


# --------------------------------------------------------------------------
# Reconstruction diagnostics -- normalized space, matching the repo's
# canonical FVE (docs/inference.md "Computing FVE — two classic footguns")
# --------------------------------------------------------------------------
MSE_SCALE = critic.mse_scale          # from the AR sidecar, never hardcoded

RECON_ROWS: list[dict] = []
# Gold activations kept per sample so the FVE denominator is computed
# on-sample. The three conditions share one h_orig, hence a separate dict.
RECON_H: dict[tuple[int, int], torch.Tensor] = {}


def recon_metrics(h_orig: torch.Tensor, h_hat: torch.Tensor) -> dict:
    """Norms, cosine and normalized MSE for one reconstruction.

    mse_nrm is the FVE numerator: MSE with BOTH sides L2-normalized to
    mse_scale, exactly what nla/loss.py trains against. It reduces to
    2(1-cos) only when mse_scale == sqrt(d_model), so compute it rather than
    deriving it. NLACritic.reconstruct() returns a RAW vector, so pred must be
    normalized here by hand (footgun #1 in docs/inference.md).

    Norms are recorded raw on purpose: normalization discards magnitude, and
    ||h_hat|| runs ~1.5x ||h_orig|| on these checkpoints, so the raw norms are
    the only place that error stays visible.
    """
    gold = h_orig.detach().float().cpu().squeeze()
    pred = h_hat.detach().float().cpu().squeeze()
    gold_n = normalize_activation(gold, MSE_SCALE)
    pred_n = normalize_activation(pred, MSE_SCALE)
    return {
        "norm_orig": gold.norm().item(),
        "norm_hat": pred.norm().item(),
        "cos": (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item(),
        "mse_nrm": ((pred_n - gold_n) ** 2).mean().item(),
    }


def log_recon(condition, h_orig, h_hat, *, n_shots, sample_idx, gold_word,
              correct, verbose=True) -> dict:
    m = recon_metrics(h_orig, h_hat)
    m.update(n_shots=n_shots, sample_idx=sample_idx, condition=condition,
             gold=gold_word, correct=bool(correct))
    RECON_ROWS.append(m)
    RECON_H[(n_shots, sample_idx)] = h_orig.detach().float().cpu().squeeze()
    if verbose:
        print(f"  [recon:{condition:<12}] ||h||={m['norm_orig']:>9.1f} "
              f"||h_hat||={m['norm_hat']:>9.1f} "
              f"ratio={m['norm_hat'] / m['norm_orig']:.3f} "
              f"cos={m['cos']:.4f} mse_nrm={m['mse_nrm']:.4f}")
    return m


RECON_COLS = ["n_shots", "sample_idx", "gold", "condition", "correct",
              "norm_orig", "norm_hat", "cos", "mse_nrm"]


def save_recon_csv(run_id: str):
    if not RECON_ROWS:
        print("[recon] nothing recorded")
        return None
    df = pd.DataFrame(RECON_ROWS)[RECON_COLS]
    path = OUT_DIR / f"recon_metrics_ioi_{run_id}.csv"
    try:
        df.to_csv(path, index=False)
        print(f"[recon] {len(df)} rows -> {path}")
    except OSError as e:
        print(f"[warn] could not save recon CSV: {e}")
        return None

    print("[recon] FVE (normalized, on-sample predict-the-mean denominator)")
    for shots, grp in df.groupby("n_shots"):
        idxs = sorted(grp["sample_idx"].unique())
        hs = [RECON_H[(shots, i)] for i in idxs if (shots, i) in RECON_H]
        if len(hs) < 2:
            continue
        _, var_nrm = compute_predict_mean_baselines(torch.stack(hs), MSE_SCALE)
        for cond, sub in grp.groupby("condition"):
            fve = 1.0 - sub["mse_nrm"].mean() / var_nrm
            print(f"  shots={shots:<3} {cond:<12} n={len(sub):<4} "
                  f"fve_nrm={fve:>8.4f}  cos={sub['cos'].mean():.4f}  "
                  f"ratio={(sub['norm_hat'] / sub['norm_orig']).mean():.3f}  "
                  f"[var_nrm={var_nrm:.4f}]")
    return path


def run_config(pairs, n_shots, rng, verbose=True) -> dict:
    baseline_correct = 0
    nla_correct = 0
    para_medium_correct = 0
    para_heavy_correct = 0
    para_medium_overlaps = []
    para_heavy_overlaps = []
    para_medium_low_overlap_count = 0
    para_heavy_low_overlap_count = 0

    # Query set is drawn from its own generator seeded identically for every
    # shot count, so all points on the sweep are measured on the SAME examples.
    # executor.py threads one rng through both query and demo selection, which
    # makes each shot count a different sample — don't inherit that.
    n = min(N_PER_CONFIG, len(pairs))
    q_rng = random.Random(SEED)
    query_idxs = q_rng.sample(range(len(pairs)), n)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"CONFIG: task=ioi  n_shots={n_shots}  n={n}")
        print(f"{'=' * 70}")

    for i, qi in enumerate(query_idxs):
        prompt, gold = build_icl_prompt(pairs, qi, n_shots, rng)
        if verbose:
            # Printed before the forward pass, so a crash still shows the exact
            # context that caused it.
            print(f"\n--- sample {i + 1}/{n}  |  gold={gold!r} ---")
            print("  [prompt]")
            print(textwrap.indent(prompt, "    | "))

        text_baseline, h_orig = run_with_patch(prompt, patch_vec=None, max_new_tokens=MAX_NEW_TOKENS)
        pred_baseline = extract_word_answer(text_baseline)
        correct_baseline = is_correct(pred_baseline, gold, case_sensitive=False)
        if verbose:
            print(f"  [baseline]  pred={pred_baseline!r:<15} correct={correct_baseline}")
        baseline_correct += correct_baseline

        explanation = av(h_orig)
        if verbose:
            print(f"  explanation : {explanation[:300]}")

        # NLA round-trip
        h_hat = ar(explanation)
        text_nla, _ = run_with_patch(prompt, patch_vec=h_hat, max_new_tokens=MAX_NEW_TOKENS)
        pred = extract_word_answer(text_nla)
        correct = is_correct(pred, gold, case_sensitive=False)
        if verbose:
            print(f"  [nla]       pred={pred!r:<15} correct={correct}")
        nla_correct += correct
        log_recon("nla", h_orig, h_hat, n_shots=n_shots, sample_idx=i,
                  gold_word=gold, correct=correct, verbose=verbose)

        # Both paraphrases in one batched generate call
        paraphrased_medium, paraphrased_heavy = paraphrase_batch(
            explanation, [PARAPHRASE_PROMPT_MEDIUM, PARAPHRASE_PROMPT_HEAVY]
        )

        # Paraphrase round-trip (medium)
        orig_words = set(explanation.lower().split())
        para_medium_words = set(paraphrased_medium.lower().split())
        overlap_medium = len(orig_words & para_medium_words) / len(orig_words) if orig_words else 0.0
        para_medium_overlaps.append(overlap_medium)
        if not is_meaningfully_paraphrased(explanation, paraphrased_medium):
            para_medium_low_overlap_count += 1
            if verbose:
                print(f"  [para-medium] WARNING: low reword rate (overlap={overlap_medium:.2f})")

        h_hat_para_medium = ar(paraphrased_medium)
        text_para_medium, _ = run_with_patch(prompt, patch_vec=h_hat_para_medium, max_new_tokens=MAX_NEW_TOKENS)
        pred_para_medium = extract_word_answer(text_para_medium)
        correct_para_medium = is_correct(pred_para_medium, gold, case_sensitive=False)
        if verbose:
            print(f"  [para-medium] text : {paraphrased_medium[:300]}")
            print(f"  [para-medium]      pred={pred_para_medium!r:<15} correct={correct_para_medium}")
        para_medium_correct += correct_para_medium
        log_recon("para_medium", h_orig, h_hat_para_medium, n_shots=n_shots,
                  sample_idx=i, gold_word=gold, correct=correct_para_medium,
                  verbose=verbose)

        # Paraphrase round-trip (heavy) -- text already generated above
        para_heavy_words = set(paraphrased_heavy.lower().split())
        overlap_heavy = len(orig_words & para_heavy_words) / len(orig_words) if orig_words else 0.0
        para_heavy_overlaps.append(overlap_heavy)
        if not is_meaningfully_paraphrased(explanation, paraphrased_heavy):
            para_heavy_low_overlap_count += 1
            if verbose:
                print(f"  [para-heavy] WARNING: low reword rate (overlap={overlap_heavy:.2f})")

        h_hat_para_heavy = ar(paraphrased_heavy)
        text_para_heavy, _ = run_with_patch(prompt, patch_vec=h_hat_para_heavy, max_new_tokens=MAX_NEW_TOKENS)
        pred_para_heavy = extract_word_answer(text_para_heavy)
        correct_para_heavy = is_correct(pred_para_heavy, gold, case_sensitive=False)
        if verbose:
            print(f"  [para-heavy] text : {paraphrased_heavy[:300]}")
            print(f"  [para-heavy]      pred={pred_para_heavy!r:<15} correct={correct_para_heavy}")
        para_heavy_correct += correct_para_heavy
        log_recon("para_heavy", h_orig, h_hat_para_heavy, n_shots=n_shots,
                  sample_idx=i, gold_word=gold, correct=correct_para_heavy,
                  verbose=verbose)

    if verbose:
        print(f"\n{'-' * 70}")
        print(f"SUMMARY  task=ioi  n_shots={n_shots}  n={n}")
        print(f"  acc_baseline (no NLA)       = {baseline_correct / n:.3f}")
        print(f"  acc_nla (round-trip)        = {nla_correct / n:.3f}")
        print(f"  acc_para_medium (paraphrase)= {para_medium_correct / n:.3f}")
        print(f"  acc_para_heavy (paraphrase) = {para_heavy_correct / n:.3f}")
        print(f"  avg_para_medium_overlap     = {sum(para_medium_overlaps) / n:.3f}")
        print(f"  avg_para_heavy_overlap      = {sum(para_heavy_overlaps) / n:.3f}")
        print(f"  para_medium_low_overlap_count = {para_medium_low_overlap_count}")
        print(f"  para_heavy_low_overlap_count  = {para_heavy_low_overlap_count}")
        print(f"{'-' * 70}\n")

    return {
        "n_shots": n_shots,
        "n": n,
        "acc_baseline": baseline_correct / n,
        "acc_nla": nla_correct / n,
        "acc_para_medium": para_medium_correct / n,
        "acc_para_heavy": para_heavy_correct / n,
        "avg_para_medium_overlap": sum(para_medium_overlaps) / n,
        "avg_para_heavy_overlap": sum(para_heavy_overlaps) / n,
        "para_medium_low_overlap_count": para_medium_low_overlap_count,
        "para_heavy_low_overlap_count": para_heavy_low_overlap_count,
    }

CONDITIONS = [
    ("acc_baseline", "o", "Baseline (no NLA)"),
    ("acc_nla", "^", "NLA round-trip"),
    ("acc_para_medium", "s", "Paraphrase round-trip (medium)"),
    ("acc_para_heavy", "d", "Paraphrase round-trip (heavy)"),
]

def plot_sweep_single(df, task_name: str, run_id: str) -> Path:
    """Per-task panel from executor.py's plot_sweep, for one task."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    for col, marker, label in CONDITIONS:
        if col not in df.columns:
            print(f"[warn] {task_name}: missing column {col!r}, skipping that line")
            continue
        ax.plot(df["n_shots"], df[col], marker=marker, label=label)
    ax.set_xlabel("Number of shots")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(task_name)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = OUT_DIR / f"fv_shot_sweep_{task_name}_{run_id}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path



if __name__ == "__main__":
    pairs = load_json_pairs(IOI_DATA)
    rng = random.Random(SEED)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"fv_shot_sweep_ioi_{run_id}.csv"

    rows = []
    for k in SHOT_COUNTS:
        rows.append(run_config(pairs, k, rng))
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        save_recon_csv(run_id)
        print(f"\n[ok] n_shots={k} -> {csv_path}")
        print(df.to_string(index=False) + "\n")
        png = plot_sweep_single(pd.DataFrame(rows), "ioi", run_id)
        print(f"[ok] plot -> {png}")

    print(f"This run's ID: {run_id}")

