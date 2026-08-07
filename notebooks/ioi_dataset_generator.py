from datasets import load_dataset
import random 
import json
import os
from pathlib import Path

USER_NAME = "abhinav"
HOME = Path("/home") / USER_NAME
NLA_REPO = Path(os.environ.get("NLA_REPO", HOME / "natural_language_autoencoders"))

ds = load_dataset(
    "fahamu/ioi", 
    data_files=["mecha_ioi_200k.parquet"]
)

ds_samples = ds['train']
rng = random.Random(0)
idx = rng.sample(range(len(ds_samples)), 200)
samples = ds['train'][idx]

with open(f"{NLA_REPO}/notebooks/datasets/ioi_1.jsonl", "w", encoding="utf-8") as f:
    for sample in samples['ioi_sentences']:
        rec = {"input": sample[:sample.rfind(' ')], "output": sample.split()[-1]}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
