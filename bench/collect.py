"""Collect Kaggle Benchmarks run outputs into one table and compare style stats with the official train set.

Usage: python bench/collect.py [bench/out] [external/own_gen_pilot.csv]
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# proxy slug -> competition label
LABEL = {
    "google/gemini-3.1-pro-preview": "gemini-3.1-pro-preview", "google/gemini-3.6-flash": "gemini-3.6-flash",
    "google/gemini-3.8-flash": "gemini-3.8-flash", "openai/gpt-5.5-2026-04-23": "gpt-5.5",
    "openai/gpt-5.6-luna": "gpt-5.6-luna", "openai/gpt-5.6-sol": "gpt-5.6-sol", "openai/gpt-5.6-terra": "gpt-5.6-terra",
    "openai/gpt-6-astra": "gpt-6-astra", "anthropic/claude-opus-4-6@default": "claude-opus-4-6",
    "anthropic/claude-opus-4-8@default": "claude-opus-4-8", "anthropic/claude-opus-5@default": "claude-opus-5",
    "anthropic/claude-sonnet-5@default": "claude-sonnet-5",
}


def style(s):
    return pd.Series({
        "n": len(s), "words_med": s.str.split().str.len().median(),
        "newline": s.str.contains("\n").mean(), "markdown": s.str.contains(r"\*\*|^#|\n#", regex=True).mean(),
        "em_dash": s.str.contains("—").mean(), "hyphen": s.str.contains(r"\w-\w", regex=True).mean(),
        "curly_apos": s.str.contains("’").mean(),
    })


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench/out"
    rows = []
    for f in out.rglob("*.run.json"):
        d = json.load(open(f))
        slug = d["modelVersion"]["slug"]
        r = (d.get("results") or [{}])[0].get("dictResult") or {}  # failed runs have no dictResult
        rows.append({"slug": slug, "model": LABEL.get(slug, "?" + slug), "state": d.get("state"),
                     "prompt_id": r.get("prompt_id"), "target_words": float(r.get("target_words", "nan")),
                     "text": r.get("text")})
    df = pd.DataFrame(rows)
    ok = df.text.notna() & (df.state == "BENCHMARK_TASK_RUN_STATE_COMPLETED")
    print(f"{len(df)} runs, {ok.sum()} with text; unknown slugs: {sorted(df.slug[df.model.str.startswith('?')].unique())}")
    df = df[ok].copy()
    df = df.sort_values("slug").drop_duplicates(["model", "prompt_id"], keep="last")  # re-runs: one per pair
    df["text_raw"] = df.text
    df["text"] = df.text.str.strip().str.replace(r"\s+", " ", regex=True)
    df["words"] = df.text.str.split().str.len()
    df["len_ratio"] = df.words / df.target_words
    refusal = df.text.str.contains(r"^(?:I can.t|I cannot|I.m sorry|Sorry|Sure|Certainly|Here is|Here.s)", regex=True)
    print(f"length ratio (words/target): median {df.len_ratio.median():.2f}; refusal/preamble: {refusal.sum()}")

    tr = pd.read_csv(ROOT / "data/train.csv")
    cmp = []
    for m, g in df.groupby("model"):
        cmp.append(style(tr.text[tr.model == m]).add_prefix("train_").rename(m))
        cmp[-1] = pd.concat([cmp[-1], style(g.text_raw).add_prefix("gen_")])
    t = pd.DataFrame(cmp)[["gen_n", "train_words_med", "gen_words_med", "gen_newline", "gen_markdown",
                           "train_em_dash", "gen_em_dash", "train_hyphen", "gen_hyphen",
                           "train_curly_apos", "gen_curly_apos"]]
    print(t.round(2).to_string())
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "external/own_gen_pilot.csv"
    df[["model", "prompt_id", "target_words", "words", "text"]].to_csv(dst, index=False)


if __name__ == "__main__":
    main()
