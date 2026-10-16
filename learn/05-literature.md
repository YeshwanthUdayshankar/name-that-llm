# 5. What the papers say

**In one sentence:** the literature agrees that LLMs leave word-level fingerprints that simple classifiers can
read, that same-family models are much harder to separate, and that accuracy keeps rising with far more data
than we have.

## Most relevant
**Sun, Yin, Xu, Kolter, Liu — "Idiosyncrasies in Large Language Models" (ICML 2025,
[arXiv 2502.12150](https://arxiv.org/abs/2502.12150))**
- 5-way ChatGPT / Claude / Grok / Gemini / DeepSeek: **97.1%** with a LoRA-tuned LLM2vec embedding model
  (rank 16, α 32, LR 5e-5, 3 epochs, mean pooling, first 512 tokens).
- **Data hunger:** accuracy converges around 10K samples per class; with 10 samples, 40%. We have 150.
- **Same family is hard:** four Qwen2.5 sizes, 4-way: 59.8%. Base vs instruct of one model: 96.8%.
- **Signal is word-level:** shuffling words keeps 88.9%; shuffling letters drops to chance; TF-IDF LR alone
  85.5%. First-word distributions differ between models.
- Takeaways for us: (1) more data is the biggest lever; (2) a LoRA-tuned LLM embedding is the strongest
  classifier family — our frozen Qwen probe is its cheap cousin; (3) our sibling confusions are expected.

**McGovern, Stureborg, Suhara, Alikaniotis — "Your Large Language Models Are Leaving Fingerprints"
(GenAIDetect @ COLING 2025, [PDF](https://aclanthology.org/2025.genaidetect-1.6.pdf))**
- Word 2–4, character 3–5 and **part-of-speech 2–4 grams** with gradient boosting match deep detectors.
- Fingerprints are consistent across domains and **within model families** (a detector trained on one LLaMA
  size recognises others; recall drops only 0.01 within family vs 0.62 across families).
- Their texts are 300–500 words; they warn short texts may carry too little signal.
- Takeaway: POS n-grams are an untried, cheap feature for us.

## Also read
- **Chen et al., EMNLP 2023** ([ACL](https://aclanthology.org/2023.emnlp-main.810)): attribution as
  next-token prediction on T5; confusion concentrates between same-family or adjacent-size models.
- **ForgePrint** ([arXiv 2609.38831](https://arxiv.org/abs/2609.38831), 2026): rewriting can push an
  attribution classifier to a chosen model (70% target success) — attribution isn't robust to adversaries.
  Not our problem: the competition passages are unmodified.
- **Code Fingerprints** ([arXiv 2603.04212](https://arxiv.org/html/2603.04212)): attribution of generated
  code; comment style is a strong signal. Different domain.
- **LLM Self-Recognition** ([2606.06315](https://www.alphaxiv.org/abs/2606.06315.md)): fingerprints injected
  at generation time; needs control of the generator, so not applicable.

## Patents
No granted patent or filing found on identifying which LLM generated a text; searches surfaced only
AI-vs-human detection papers. A proper search would use Google Patents / USPTO / EPO with terms like "model
attribution", "source model identification", "generative model fingerprinting".
