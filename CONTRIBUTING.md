# Contributing

Contributions welcome! Open a PR to add papers, reports, or tools related to on-policy distillation.

## Inclusion criteria

A paper must pass **one** of these two tests:

1. **Student rollouts are central to the learning signal.** The student generates its own outputs, and the teacher (external, self, or privileged) provides supervision on those outputs. Student rollouts must be the primary training substrate, not incidental to the method.
2. **The work directly enables OPD deployment.** Cross-tokenizer methods, training frameworks, surveys, and tooling that make OPD practical.

### What does NOT qualify

- **Survey classification alone.** "The survey classifies it as OPD" is not sufficient. The paper's own mechanism must pass the test above.
- **Near-policy is not on-policy.** Methods using historical checkpoints or past model versions generate from a different policy than the one being trained. This is near-policy, not on-policy.
- **Self-play is not distillation.** Adversarial self-play (e.g., SPIN) involves student rollouts but lacks teacher-to-student knowledge transfer. It may fit in Precursors but not in Core OPD.
- **Reframing is not enough.** Casting alignment, imitation learning, or RL as "policy distillation" in the abstract does not make student rollouts operationally central. Check the actual training loop.
- **Domain-specific applications** need a transferable methodological contribution beyond the domain. A paper that applies vanilla OPD to a new dataset does not clear the bar.

### When in doubt

Read the actual paper. One-line descriptions and survey classifications are not sufficient for borderline decisions. If the verdict depends on whether the student generates its own training data in the core training loop, check the method section.

## Modality fairness

Apply the same bar to multimodal, speech, agent, and embodied papers as to text-only ones. If a speech-LLM paper uses student on-policy rollouts with token-level teacher feedback, it clears the same criterion as a text-only paper with the same structure. Do not apply an implicit "narrowness" penalty to non-text modalities.

## Section placement

The README's top-level sections, in order. Place an entry by matching its primary contribution.

### Surveys and Essays

Surveys, position papers, and analyses (**Surveys and Position Papers**), plus blog posts, talks, and walkthroughs (**Essays, Blog Posts, and Walkthroughs**).

### Core OPD Papers (selective canonical entries)

Only add here if the paper's primary contribution is a new component of the OPD training loop — an objective, teacher-access variant, self-distillation variant, stability fix, context-internalization mechanism, or systems/efficiency/privacy constraint — with student rollouts central to the learning signal and evaluated on LLM text generation or reasoning. If removing the OPD-loop component leaves a working contribution (an RL recipe, preference method, or KD baseline), it is Adjacent. When unsure, place in Adjacent.

### Adjacent and Enabling Work

Work that is not canonical OPD but matters for understanding or deploying it. Subsections:

- **Cross-Tokenizer and Model-Family Enablers** — vocabulary alignment, cross-architecture methods
- **Mismatch Mitigation and Student Quality** — distribution gap, adaptive mixing, student output quality
- **Preference, Reward-Guided, and Hybrid RL+KD** — preference optimization, reward-guided distillation, RL+KD unification
- **Self-Play and Iterative Bootstrapping** — SPIN-style self-play, iterative best-of-N, MCTS-guided self-evolution
- **Precursors** — historical antecedents that predate the OPD label

### Domain Extensions

OPD applied to substrates beyond LLM text reasoning — student rollouts are still central, but the substrate is the point. Subsections by substrate:

- **Agents and Tool-Use** — multi-turn agents, tool use, search, GUI, RAG, memory
- **Multimodal and Vision-Language** — VLMs/MLLMs, visual and video reasoning
- **Speech and Audio** — ASR, speech LLMs, audio reasoning
- **Diffusion, Flow, and Generative Media** — diffusion/flow image and video generation, diffusion language models
- **Embodied, Robotics, and Control** — VLA, robotic manipulation, driving, world models
- **Speculative Decoding (Draft-Model Training)** — DistillSpec, EAGLE, HASS families; draft trained from target supervision on its own rollouts

A method whose contribution is a transferable OPD mechanism belongs in Core even when evaluated on an agent or multimodal task; place it in Domain Extensions only when the substrate is the point.

### Technical Reports and Industrial Recipes

Production systems using OPD as a post-training stage. Table format.

### Frameworks and Implementations

Training frameworks (**Training Frameworks**) and single-method or reference code (**Implementations**). Table or bullet format depending on subsection.

## Entry format

All entries are single-line. The title is the hyperlink; the year is italicized; the description follows an em dash:

```
- [Full Paper Title](url) *(Year)* — One-line description.
```

For arxiv papers, link directly to `https://arxiv.org/abs/XXXX.XXXXX`. For blog posts, essays, model cards, or repos, use the canonical URL.

If the entry has an associated code release, append a parenthesized Code link at the end:

```
- [Title](arxiv-url) *(Year)* — Description. ([Code](repo-url))
```

For implementation repos under `### Implementations`, omit the year: `- [Repo Name](url) — Description.`

**Tables** (Technical Reports, Training Frameworks) follow the column layout already in each section — match the existing rows.

The description must be a single sentence, **hard cap 22 words** (well-written entries average 12–16). State **one mechanism phrase plus one differentiator** — how the method works at the highest level, then what makes it distinct from the rest of the list. If you need a second mechanism clause to explain the first, the first was wrong. Forbidden: equation symbols, Greek letters, benchmark numbers, hyperparameters, model/dataset enumerations, ablation lists, percentage-point gains, and any second mechanism clause introduced by a semicolon or "and." **Deletion test before committing:** at each comma, semicolon, or "and" in your draft, try deleting the trailing clause; if the entry still tells a scanning reader why to click, that clause was bloat. Look at existing entries for calibration; if your draft is longer than the longest entry in the same subsection, restart.

## Batch additions

When adding multiple papers at once:

- **Relevance is the bar.** Every paper meeting the inclusion criteria goes in, however crowded its area. Do not withhold a qualifying paper as "marginal" — a fourth variant in a well-covered niche still belongs on the list.
- **Gap-filling orders, it does not gate.** A paper opening a new niche (first speech OPD, first embodied OPD) is worth leading with, but nothing qualifying is dropped for ranking below it.
- **Flag section growth.** If a single update would double any subsection's size, say so — that signals the section may need splitting, not that entries should be cut.

## Taxonomy and reading path

- Update the taxonomy tables only for Core OPD papers or papers that clearly extend an existing taxonomy row.
- Method names in the taxonomy tables are markdown links to the paper's URL: `[ShortName](url)`, comma-separated, reusing the same URL as the paper's list entry.
- The Start Here reading path changes rarely. Only add a paper if it is the best introduction to a topic currently underrepresented in the path.
