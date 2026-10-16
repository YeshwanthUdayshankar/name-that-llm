# 1. How the competition works

**In one sentence:** given a 40–250 word paragraph, predict which of 18 frontier LLMs wrote it; Kaggle runs our
notebook offline on ~4,000 hidden passages and scores balanced accuracy.

## The task
- **18 models from 6 labs:** OpenAI (`gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`),
  Google (`gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`),
  Anthropic (`claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`),
  ZAI (`glm-5.3`, `glm-5.3-flash`), Qwen (`qwen3.8-max`), Moonshot (`kimi-k3`).
- **Passages** are self-contained answers on philosophy, physics, maths, biology, computer science and creative
  prose. Length varies randomly (230–1,900 characters, median 670; at most 351 DeBERTa tokens).

## The data
- `train.csv`: 2,699 passages = 150 prompts × 18 models (one prompt has 17). Columns `id, prompt_id, text, model`.
  **Every prompt is answered exactly once by every model** — this turned out to matter a lot ([guide 4](04-prompt-structure-assignment.md)).
- `test.csv`: 3 example rows while we develop; ~4,000 passages in the hidden re-run. **Test prompts never appear
  in train.** Per-model counts are unequal, and the model mix differs between the public and private leaderboards.

## The metric: balanced accuracy
The average over the 18 models of per-model recall. Every model counts equally however many test passages it
has, so a rare model's mistakes cost as much as a common one's. Random guessing scores 1/18 = 0.056.

## How a submission works (code competition)
1. We push a notebook (`kaggle kernels push -p kaggle/<dir>`). It reads
   `/kaggle/input/competitions/name-that-llm/test.csv` and writes `submission.parquet` (`id, model`).
2. Kaggle runs it once on the 3 example rows (the "check run").
3. On submit, Kaggle re-runs it on the hidden ~4,000 rows and scores it.

Limits: CPU or GPU notebook ≤ 9 h, **internet disabled**, public external data and pretrained models allowed,
5 submissions/day, 2 final picks, teams up to 5. Final deadline **2027-01-13 23:59 UTC** (6:59 PM ET).
Prizes: $5,000 each to the top 10.

## Rules that constrain us
- **Internet must be off** for a submitted notebook (the toggle exists, but the Submit button requires it off).
  Models must be attached as Kaggle inputs — Kaggle hosts the official Qwen 3 checkpoints, so no upload needed.
- **No private sharing of code or data outside the team.** Public sharing must happen on Kaggle's forums or
  notebooks. So our GitHub repo is **private**, and competition data never goes into git.
- **No hand-labelling** of test data. Automated use of test text (clustering, transductive tricks) is fine.

## Budget
$15/week plus a $5 buffer, $20 max (shared with other projects). So far this project has cost **$0**: local RTX
3070 Ti (8 GB) for experiments, Kaggle's free GPU quota (30 h/week) for submissions. The only paid lever
considered is generating extra training data through the 18 models' APIs, which needs a priced decision first.
