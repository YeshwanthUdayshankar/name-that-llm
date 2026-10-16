# 3. Features: what worked and what didn't

**In one sentence:** with only 150 passages per model, frozen LLM hidden states plus a simple linear classifier
beat fine-tuning, and blending them with TF-IDF n-grams adds ~5 points; deeper structure (pooling tricks,
hierarchies) didn't help.

## Results (5-fold, grouped by prompt)
| approach | balanced acc. |
|---|---|
| TF-IDF word 1–2 + char 1–5 grams, LinearSVC / LogisticRegression | 0.334 / 0.338 |
| DeBERTa-v3-base full fine-tune (fold 0 only; baseline fold 0 = 0.313) | 0.252 |
| Qwen3-0.6B-Base frozen, layers 4 + 20 | 0.393 |
| Qwen3-1.7B-Base frozen, layers 4 + 16 | 0.407 |
| Qwen3-4B-Base frozen (first 24 of 36 layers), layers 4 + 8 + 20 | 0.436 |
| blend TF-IDF .5 + 0.6B .1 + 4B .4 (v2) | 0.458 (equal weights 0.451) |

## TF-IDF baseline (`src/baseline.py`)
Word 1–2 grams and character 1–5 grams, **case and punctuation kept** (style lives there), sublinear TF.
It gets the lab right 69% of the time but the exact model only 33%. Worst confusions are siblings:
`gemini-3.8-flash → 3.7-flash` (56 of 150), `gpt-5.6-sol → luna`, and `glm-5.3`, `glm-5.3-flash`,
`claude-fable-5` all collapsing into `claude-opus-5`.

## Why fine-tuning DeBERTa failed
- First attempt: **loss = nan from step one.** transformers 5.x loads weights in the checkpoint's dtype (fp16
  for DeBERTa-v3), and AdamW updates on fp16 weights blow up. Fix: `from_pretrained(..., dtype=torch.float32)`.
- Even then it reached only 0.252 on fold 0 (5 epochs) and 0.230 (10 epochs, higher LR — unstable).
  ~2,200 training passages is too few to fine-tune 184M parameters for an 18-way style task.

## Why frozen LLM features work (`src/embed.py`, `src/probe.py`)
Run a base LLM over each passage without training it, average each layer's hidden states over the tokens, and
fit `StandardScaler + LogisticRegression(C=0.05)` on top. Only the small linear head learns from our data.

- **Early layers carry style.** Qwen3-0.6B per-layer scores: layer 4 = 0.365, falling to 0.292 at layer 28.
  Early layers encode surface form (word choice, punctuation); late layers encode meaning, which is shared
  across models answering the same prompt.
- **One early + one middle layer** beats any single layer (0.6B: 4+20 = 0.393 vs 4 alone = 0.365).
- **Bigger models help:** 0.6B 0.393 → 1.7B 0.407 → 4B 0.436.
- **Log-probability statistics** (how predictable the text is to Qwen) are weak alone: 0.11.
- **Base models**, not instruct: we want a neutral reader of style.

## Blending (`src/blend.py`)
Weighted average of log-probabilities. TF-IDF and LLM features make different mistakes, so the blend gains
~2–5 points over the best single model.

## Dead ends
- **Hierarchical lab → model classifier** (`src/hier.py`): 0.456, same as a flat classifier with identical
  features (0.456). Lab accuracy rose (0.757 → 0.783) but only for big labs; small labs got worse
  (GLM 0.46 → 0.39, Kimi 0.48 → 0.35). Same features ⇒ the specialists had nothing new to learn from.
- **Extra token pooling** (`src/pool_probe.py`): mean = 0.435; adding max 0.415, std 0.429, last-8 tokens
  0.410, first-8 tokens 0.379, all five 0.394. More dimensions on 150 samples/class just overfit.

## Where the errors are (v2 blend, OOF)
Lab right 76% of the time; within the right lab, right model 60%. Even a perfect lab classifier would only
reach 0.598. The small labs (GLM, Kimi, Qwen) are recognised only ~half the time.
