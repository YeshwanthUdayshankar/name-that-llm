"""Daily free generation run on the Kaggle Benchmarks Model Proxy (about $10/day free quota).

Once per UTC day (retried hourly until the free quota lets calls through): take the next day of prompts from prompts_bank.csv, push them as task ntl-gen-daily, run the
12 proxy models in small batches (all at once trips 403 quota reservations), download, collect every run so far
into external/own_gen.csv, and run the transfer check. Safe to call hourly from cron: it exits if today's day
has already started, and a lock stops overlapping runs. State and logs live in bench/out/ (git-ignored).

Usage: python bench/daily.py [--force]   (--force: run even if a day already ran today)
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "bench"
OUT = B / "out"
STATE = OUT / "daily_state.json"
TASK = "ntl-gen-daily"
KAGGLE = os.path.expanduser("~/.local/bin/kaggle")
PY = str(ROOT / ".venv/bin/python")
# Pushing a task already runs it on gemini-3.8-flash (the CLI's default model), so it is not in the batches.
# Gemini 3.1 Pro alone uses about half the quota (long hidden reasoning), so it gets a batch to itself, last.
BATCHES = [["gemini-3.6-flash", "gpt-5.5-2026-04-23", "gpt-6-astra"],
           ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"],
           ["claude-opus-4-6-default", "claude-opus-4-8-default", "claude-opus-5-default", "claude-sonnet-5-default"],
           ["gemini-3.1-pro-preview"]]

TASK_SRC = '''# Name That LLM: daily style-matched passages (day {day}). Prompts from bench/prompts_bank.csv; one user
# message, no system prompt, provider-default sampling and reasoning, same wrapper as the pilot.
import kaggle_benchmarks as kbench
import pandas as pd

ROWS = {rows}


@kbench.task(name="{task}", description="Generate one single-paragraph passage per prompt.")
def ntl_gen_daily(llm, prompt_id: str, prompt: str, target_words: int) -> dict:
    # max_tokens caps the proxy's up-front cost reservation; 4000 leaves room for hidden reasoning tokens.
    with kbench.chats.new(prompt_id) as chat:
        text = llm.prompt(f"{{prompt}} Write a single paragraph of about {{target_words}} words, with no title or headings.",
                          extra_api_params={{"max_tokens": 4000}})
        usage = chat.usage
    kbench.assertions.assert_not_empty(text.strip(), expectation="non-empty passage")
    return {{"prompt_id": prompt_id, "target_words": target_words, "text": text,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}}

runs = ntl_gen_daily.evaluate(llm=[kbench.llm], evaluation_data=pd.DataFrame(ROWS), n_jobs=2, timeout=300,
                              max_attempts=3, retry_delay=15)
'''


def log(msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}"
    print(line, flush=True)
    with open(OUT / "daily.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout, keep=6):
    r = subprocess.run(cmd, cwd=B, capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout + r.stderr).strip().splitlines()[-keep:]
    return r.returncode, "\n    ".join(tail)


def wait_ready(limit_s=3600):
    """`tasks run` refuses while the task is still running (e.g. the push-triggered run)."""
    for _ in range(limit_s // 30):
        rc, out = sh([KAGGLE, "benchmarks", "tasks", "status", TASK], 120, keep=999)
        st = next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.strip().startswith("Status:")), "")
        if st and st not in ("Running", "Queued", "Pending", "Scheduled"):
            return st
        time.sleep(30)
    return "timeout"


def main():
    OUT.mkdir(exist_ok=True)
    lock = open(OUT / "daily.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {"days": {}}
    today = f"{datetime.now(timezone.utc):%Y-%m-%d}"
    if "--force" not in sys.argv and any(d["date"] == today for d in state["days"].values()):
        return
    bank = pd.read_csv(B / "prompts_bank.csv")
    day = max(map(int, state["days"]), default=0) + 1
    rows = bank[bank.day == day]
    if rows.empty:
        log(f"prompt bank exhausted after day {day - 1}; add topics and rerun build_prompts.py")
        return
    state["days"][str(day)] = {"date": today, "status": "started", "n_prompts": len(rows)}
    STATE.write_text(json.dumps(state, indent=1))
    log(f"day {day}: {len(rows)} prompts")

    recs = rows[["prompt_id", "prompt", "target_words"]].to_dict("records")
    (B / "ntl_gen_daily.py").write_text(TASK_SRC.format(day=day, task=TASK, rows=json.dumps(recs, indent=1)))
    rc, out = sh([KAGGLE, "benchmarks", "tasks", "push", TASK, "-f", "ntl_gen_daily.py"], 900)
    log(f"push rc={rc}\n    {out}")
    if rc:
        state["days"][str(day)]["status"] = "push_failed"
        STATE.write_text(json.dumps(state, indent=1))
        return
    for models in BATCHES:
        log(f"task status before batch: {wait_ready()}")
        cmd = [KAGGLE, "benchmarks", "tasks", "run", TASK, "--wait", "5400"]
        for m in models:
            cmd += ["-m", m]
        try:
            rc, out = sh(cmd, 6000)
        except subprocess.TimeoutExpired:
            rc, out = -1, "timed out"
        log(f"run {models} rc={rc}\n    {out}")
    wait_ready()
    rc, out = sh([KAGGLE, "benchmarks", "tasks", "download", TASK, "-o", str(OUT)], 900)
    log(f"download rc={rc}")

    rc, out = sh([PY, "-I", str(B / "collect.py"), str(OUT), str(ROOT / "external/own_gen.csv")], 600)
    log(f"collect rc={rc}\n    {out}")
    g = pd.read_csv(ROOT / "external/own_gen.csv")
    got = g[g.prompt_id.str.startswith(f"d{day:02d}_")].model.value_counts()
    missing = {m: int(len(rows) - got.get(m, 0)) for m in sorted(set(g.model)) if got.get(m, 0) < len(rows)}
    log(f"day {day} passages: {int(got.sum())}/{len(rows) * 12}; short models: {missing or 'none'}; "
        f"total own passages: {len(g)}")
    if not got.sum():  # every call refused: the free quota is a rolling window, not a UTC-midnight reset
        del state["days"][str(day)]
        STATE.write_text(json.dumps(state, indent=1))
        log(f"day {day}: no passages (quota not free yet?); will retry next hour")
        return
    # transfer only (one TF-IDF fit), memory-capped so it cannot take down the desktop
    rc, out = sh(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=12G", "-p", "MemorySwapMax=0",
                  PY, "-I", str(ROOT / "src/ext_check2.py"), "own", "--transfer-only"], 3600)
    log(f"transfer check rc={rc}\n    {out}")
    state["days"][str(day)].update(status="done", passages=int(got.sum()), missing=missing)
    STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
