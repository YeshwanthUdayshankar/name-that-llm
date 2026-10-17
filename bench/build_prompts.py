"""Build the daily prompt bank from topics.txt: one templated prompt per topic, split into days of PER_DAY.

Templates follow the reconstructed train protocol (explain / describe / argue / story, with tone adjectives).
Target lengths are stratified quantiles of the train word counts, so each day matches the train length
distribution (the pilot's 30 random draws came out short, median 73 vs 101).

Usage: python bench/build_prompts.py -> bench/prompts_bank.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PER_DAY = 70
ADJ = ["vivid", "evocative", "poetic", "lyrical", "reflective", "short", "engaging", "sensory"]
EXPLAIN = ["Explain {t}.", "In simple, everyday terms for a general audience, explain {t}.",
           "Using a vivid analogy, explain {t}.", "Write a short passage explaining {t}.",
           "In engaging prose for a curious reader with no background in {d}, explain {t}."]
DESCRIBE = ["Write a {a} passage describing {t}.", "Describe {t} in {a} prose."]
ARGUE = ["Write a reflective passage on what {t} reveals about how we think.",
         "Write a short persuasive passage arguing that more people should know about {t}."]
STORY = ["Write a {a} short story about {t}.", "Write a {a} vignette about {t}.", "Write a {a} passage describing {t}."]


def main():
    rng = np.random.default_rng(2027)
    rows = []
    for line in open(ROOT / "bench/topics.txt"):
        if line.strip() and not line.startswith("#"):
            d, t = (x.strip() for x in line.split("|", 1))
            rows.append((d, t))
    order = rng.permutation(len(rows))
    wc = np.sort(pd.read_csv(ROOT / "data/train.csv").text.str.split().str.len().values)
    out = []
    for j, i in enumerate(order):
        d, t = rows[i]
        if d == "creative prose":
            kind, tpl = "story", rng.choice(STORY)
        else:
            u = rng.random()
            argue_p = 0.25 if d in ("philosophy", "economics", "psychology") else 0.05
            kind = "argue" if u < argue_p else ("describe" if u < argue_p + 0.3 else "explain")
            tpl = rng.choice({"argue": ARGUE, "describe": DESCRIBE, "explain": EXPLAIN}[kind])
        a = rng.choice(ADJ)
        prompt = tpl.format(t=t, d=d if d != "other" else "the subject", a=a)
        prompt = prompt[0].upper() + prompt[1:]
        out.append({"day": j // PER_DAY + 1, "domain": d, "task_type": kind, "prompt": prompt})
    df = pd.DataFrame(out)
    tw = np.zeros(len(df), int)
    for day, g in df.groupby("day"):  # stratified length quantiles within each day
        q = wc[((np.arange(len(g)) + rng.random(len(g))) / len(g) * (len(wc) - 1)).astype(int)]
        tw[g.index] = rng.permutation(q)
    df["target_words"] = tw
    df.insert(0, "prompt_id", [f"d{r.day:02d}_{k:02d}" for k, r in zip(df.groupby("day").cumcount(), df.itertuples())])
    df.to_csv(ROOT / "bench/prompts_bank.csv", index=False)
    print(df.groupby("day").agg(n=("prompt", "size"), words_med=("target_words", "median")).T.to_string())
    print(df.task_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
