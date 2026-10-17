# Name That LLM: one-off test, the 30 pilot prompts with reasoning="low" (Geminis only). Question: does lower
# hidden reasoning keep each model's style? If so the daily run can stretch the free quota.
# Copied from ntl_gen_pilot.py.
# Prompts are new (checked against the 150 reconstructed train topics); target lengths are drawn from the
# competition train word-count distribution. One user message, no system prompt, provider-default sampling.
import kaggle_benchmarks as kbench
import pandas as pd

ROWS = [
 {
  "prompt_id": "g001",
  "prompt": "Explain the Mpemba effect, the claim that hot water can sometimes freeze faster than cold water, and why physicists still argue about whether it is real.",
  "target_words": 62
 },
 {
  "prompt_id": "g002",
  "prompt": "Using an everyday analogy, explain how a laser produces light in which every wave marches in step, through stimulated emission and population inversion.",
  "target_words": 53
 },
 {
  "prompt_id": "g003",
  "prompt": "Describe, vividly and for a general audience, how a tokamak confines a plasma hotter than the core of the Sun using twisted magnetic fields.",
  "target_words": 146
 },
 {
  "prompt_id": "g004",
  "prompt": "Explain why the sky glows red and orange at sunset, tracing how Rayleigh scattering changes as sunlight crosses more of the atmosphere.",
  "target_words": 121
 },
 {
  "prompt_id": "g005",
  "prompt": "Explain the Casimir effect, in which two uncharged metal plates in a vacuum are pushed together, and what it suggests about empty space.",
  "target_words": 40
 },
 {
  "prompt_id": "g006",
  "prompt": "Explain how a sonic boom forms when an aircraft breaks the sound barrier, and why the boom follows the plane rather than happening once.",
  "target_words": 43
 },
 {
  "prompt_id": "g007",
  "prompt": "Explain how the arctic tern migrates from pole to pole every year, seeing more daylight than any other animal, and how it navigates.",
  "target_words": 50
 },
 {
  "prompt_id": "g008",
  "prompt": "Describe in a whimsical tone how a cuckoo chick tricks its foster parents into feeding it while it pushes their own eggs out of the nest.",
  "target_words": 161
 },
 {
  "prompt_id": "g009",
  "prompt": "Explain how the immortal jellyfish Turritopsis dohrnii can revert to an earlier stage of its life cycle, and what that does and does not mean for ageing.",
  "target_words": 50
 },
 {
  "prompt_id": "g010",
  "prompt": "Explain quorum sensing, the way bacteria count their own numbers with chemical signals and switch on behaviours like glowing only when crowded.",
  "target_words": 137
 },
 {
  "prompt_id": "g011",
  "prompt": "Write a haunting description of the deep-sea anglerfish, its glowing lure and the strange way males fuse permanently to females.",
  "target_words": 53
 },
 {
  "prompt_id": "g012",
  "prompt": "Make the case, in a reflective tone, that Pascal's wager fails once you notice how many possible gods there could be.",
  "target_words": 62
 },
 {
  "prompt_id": "g013",
  "prompt": "Explain the problem of moral luck, using the example of two equally careless drivers where only one happens to hit a child.",
  "target_words": 59
 },
 {
  "prompt_id": "g014",
  "prompt": "Explain the sorites paradox, asking how many grains of sand you can remove before a heap stops being a heap, and what it says about vague words.",
  "target_words": 61
 },
 {
  "prompt_id": "g015",
  "prompt": "Explain Nick Bostrom's paperclip maximizer thought experiment and the worry about AI goals that it is meant to illustrate.",
  "target_words": 40
 },
 {
  "prompt_id": "g016",
  "prompt": "Write a short, lyrical story about a lighthouse keeper who discovers that the light has been signalling to something out at sea.",
  "target_words": 42
 },
 {
  "prompt_id": "g017",
  "prompt": "Write a poetic description of a city during the first heavy snowfall of the year, seen from the window of a night train.",
  "target_words": 124
 },
 {
  "prompt_id": "g018",
  "prompt": "Write a melancholy vignette about the last speaker of a language recording a lullaby for a linguist.",
  "target_words": 187
 },
 {
  "prompt_id": "g019",
  "prompt": "Explain Benford's law, the surprising fact that in many real-world datasets the leading digit 1 appears about 30% of the time, and how auditors use it to spot fraud.",
  "target_words": 104
 },
 {
  "prompt_id": "g020",
  "prompt": "Explain the four colour theorem, which says any map needs at most four colours so neighbours differ, and why its 1976 computer-assisted proof was controversial.",
  "target_words": 49
 },
 {
  "prompt_id": "g021",
  "prompt": "Explain the birthday paradox, showing why only 23 people are needed for a better-than-even chance that two share a birthday.",
  "target_words": 175
 },
 {
  "prompt_id": "g022",
  "prompt": "Describe the Great Molasses Flood of 1919 in Boston, when a storage tank burst and a wave of syrup swept through the streets.",
  "target_words": 90
 },
 {
  "prompt_id": "g023",
  "prompt": "Describe the Tunguska event of 1908, when an explosion flattened 80 million trees across the Siberian forest without leaving a crater.",
  "target_words": 124
 },
 {
  "prompt_id": "g024",
  "prompt": "Describe the Dancing Plague of 1518 in Strasbourg, when hundreds of people danced for days, and the theories offered to explain it.",
  "target_words": 70
 },
 {
  "prompt_id": "g025",
  "prompt": "Explain public-key cryptography using the analogy of a padlock that anyone can close but only one person can open.",
  "target_words": 105
 },
 {
  "prompt_id": "g026",
  "prompt": "Explain how a Bloom filter can tell you that an item is definitely not in a set, or only probably in it, while using very little memory.",
  "target_words": 55
 },
 {
  "prompt_id": "g027",
  "prompt": "Explain the Dunning-Kruger effect and why people with the least skill in an area are often the most confident about it.",
  "target_words": 122
 },
 {
  "prompt_id": "g028",
  "prompt": "Explain the spotlight effect, our tendency to overestimate how much other people notice our appearance and mistakes.",
  "target_words": 76
 },
 {
  "prompt_id": "g029",
  "prompt": "Explain the lipstick effect, the idea that consumers keep buying small luxuries during recessions, and why economists debate whether it is real.",
  "target_words": 99
 },
 {
  "prompt_id": "g030",
  "prompt": "Explain the cobra effect, named after a colonial bounty on snakes in Delhi, as an example of incentives that backfire.",
  "target_words": 82
 }
]

@kbench.task(name="ntl-gen-lowthink", description="Generate one single-paragraph passage per prompt.")
def ntl_gen_lowthink(llm, prompt_id: str, prompt: str, target_words: int) -> dict:
    # max_tokens caps the proxy's up-front cost reservation (the default reserves the model's full output
    # limit and fails with 403 on the free quota); 4000 leaves room for hidden reasoning tokens.
    with kbench.chats.new(prompt_id) as chat:
        text = llm.prompt(f"{prompt} Write a single paragraph of about {target_words} words, with no title or headings.",
                          reasoning="low", extra_api_params={"max_tokens": 4000})
        usage = chat.usage
    kbench.assertions.assert_not_empty(text.strip(), expectation="non-empty passage")
    return {"prompt_id": prompt_id, "target_words": target_words, "text": text,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}

runs = ntl_gen_lowthink.evaluate(llm=[kbench.llm], evaluation_data=pd.DataFrame(ROWS), n_jobs=2, timeout=300,
                                  max_attempts=3, retry_delay=15)
