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

### Core OPD Papers (~21 canonical entries)

Only add here if the paper's primary contribution is a new OPD method and it would be cited as a defining work in the field. When unsure, place in Adjacent.

### Adjacent and Enabling Work

The default landing zone. Subsections:

- **Cross-Tokenizer and Model-Family Enablers** — vocabulary alignment, cross-architecture methods
- **Mismatch Mitigation and Student Quality** — distribution gap, adaptive mixing, student output quality
- **Preference, Reward-Guided, and Hybrid RL+KD** — preference optimization, reward-guided distillation, RL+KD unification
- **Agent Distillation, Multimodal, and Other Extensions** — agents, VLMs, speech, embodied, other modalities
- **Precursors** — historical antecedents that predate the OPD label

### Technical Reports

Production systems using OPD as a post-training stage. Table format.

### Frameworks and Tools

Training frameworks and code. Table or bullet format depending on subsection.

## Entry format

```
- [Full Paper Title](url) (Year) — One-line description.
```

The description should be a single terse sentence capturing what makes this paper distinctive. Look at existing entries for calibration.

## Batch additions

When adding multiple papers at once:

- **Flag section growth.** If a single update would double any subsection's size, consider whether the section needs splitting or whether some entries are marginal.
- **Prioritize gap-filling over completeness.** A paper that opens a new niche (first speech OPD, first embodied OPD) has higher priority than a fourth variant in a well-covered area.
- **Cap awareness.** Large batches dilute curation signal. Prefer 5-8 high-confidence additions over 12+ with several borderline entries.

## Taxonomy and reading path

- Update the taxonomy tables only for Core OPD papers or papers that clearly extend an existing taxonomy row.
- The Start Here reading path changes rarely. Only add a paper if it is the best introduction to a topic currently underrepresented in the path.
