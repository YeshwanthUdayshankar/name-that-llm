# Learn: the ideas behind this project

Plain-language guides to what we ran into while competing in Kaggle's
["Name That LLM"](https://www.kaggle.com/competitions/name-that-llm) (identify which of 18 LLMs wrote a short
passage). Each guide explains the idea, shows where it mattered here (with our own numbers), and lists the
papers or pages we used. Read them in order the first time.

| # | guide | what you'll understand afterwards |
|---|---|---|
| 1 | [How the competition works](01-how-the-competition-works.md) | the task, data, metric, notebook-only submissions, the rules that constrain us (internet off, no private sharing), our budget |
| 2 | [Validation and the leaderboard](02-validation-and-the-leaderboard.md) | why we split by prompt, how well local scores predict the leaderboard, leaderboard noise, optimistic blend weights |
| 3 | [Features: what worked and what didn't](03-features-what-worked.md) | TF-IDF, the failed DeBERTa fine-tune (and the fp16 bug), frozen Qwen3 hidden states, which layers, model size, blending, and two dead ends |
| 4 | [Exploiting the prompt structure](04-prompt-structure-assignment.md) | each model answers each prompt once; clustering passages by topic; the Hungarian assignment; simulations and risks |
| 5 | [What the papers say](05-literature.md) | the attribution literature that matters here and what we took from each paper |
| 6 | [Engineering notes](06-engineering-notes.md) | running Qwen offline on Kaggle, fitting big models in 8 GB, T4 precision, Kaggle CLI quirks |
| 7 | [Strategy](07-strategy.md) | where we stand, what to try next and in what order, budget |
| 8 | [Lessons learned](08-lessons-learned.md) | the short list of things we'd tell ourselves on day one |

Live state: [README.md](../README.md) (results table) and the scripts in `src/` and `kaggle/`.
