# ⚗️ Awesome On-Policy Distillation

<p align="center">
  <a href="https://awesome.re"><img src="https://img.shields.io/badge/Awesome-%E2%9A%97%EF%B8%8F_On--Policy_Distillation-000000?style=for-the-badge&labelColor=000000" alt="Awesome On-Policy Distillation"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/stargazers"><img src="https://img.shields.io/github/stars/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Stars&labelColor=000000&color=000000" alt="GitHub Stars"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/network/members"><img src="https://img.shields.io/github/forks/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Forks&labelColor=000000&color=000000" alt="GitHub Forks"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/commits"><img src="https://img.shields.io/github/last-commit/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Last%20Commit&labelColor=000000&color=000000" alt="Last Commit"></a>
</p>

A curated collection of papers, technical reports, frameworks, and tools for on-policy distillation (OPD) of large language models.

> **On-policy distillation** trains a student on samples from its own evolving policy, while a teacher (external, privileged, or self-conditioned) provides dense supervision on those same samples.

OPD sits between supervised fine-tuning and reinforcement learning. Unlike off-policy KD, the student trains on its *own* generations, closing the train-inference distribution gap. Unlike RL, the student receives dense token-level teacher guidance rather than sparse rewards.

As of 2026, OPD is a standard post-training primitive at Alibaba (Qwen3), Xiaomi (MiMo), Zhipu (GLM-5), NVIDIA (Nemotron-Cascade 2), and others.

## Contents

- [Quick Start by Role](#quick-start-by-role)
- [Start Here](#start-here)
- [Surveys](#surveys)
- [Taxonomy](#taxonomy)
  - [By Teacher Type](#by-teacher-type)
  - [By Primary Goal](#by-primary-goal)
- [Core OPD Papers](#core-opd-papers)
  - [Foundations](#foundations)
  - [Gap-Bridging](#gap-bridging)
  - [Stability and Objective Design](#stability-and-objective-design)
  - [Self-Distillation](#self-distillation)
  - [Context and Experience Internalization](#context-and-experience-internalization)
  - [Efficiency Variants](#efficiency-variants)
- [Adjacent and Enabling Work](#adjacent-and-enabling-work)
  - [Cross-Tokenizer and Model-Family Enablers](#cross-tokenizer-and-model-family-enablers)
  - [Mismatch Mitigation and Student Quality](#mismatch-mitigation-and-student-quality)
  - [Preference, Reward-Guided, and Hybrid RL+KD](#preference-reward-guided-and-hybrid-rlkd)
  - [Self-Play and Iterative Bootstrapping](#self-play-and-iterative-bootstrapping)
  - [Agent Distillation, Multimodal, and Other Extensions](#agent-distillation-multimodal-and-other-extensions)
  - [Speculative Decoding (Draft-Model Training)](#speculative-decoding-draft-model-training)
  - [Precursors](#precursors)
- [Technical Reports and Industrial Recipes](#technical-reports-and-industrial-recipes)
- [Frameworks, Tools, and Implementations](#frameworks-tools-and-implementations)
  - [Training Frameworks](#training-frameworks)
  - [Code, Tutorials, and Guides](#code-tutorials-and-guides)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [Citation](#citation)

## Quick Start by Role

| Role | Start With | Key Resources |
|---|---|---|
| New to distillation | Definition above, then the [reading path](#start-here) | [Start Here](#start-here) |
| Researcher surveying the field | [Core OPD Papers](#core-opd-papers) for the canonical 21 | [Taxonomy](#taxonomy) for a structured map |
| Building an OPD pipeline | [TRL](https://huggingface.co/docs/trl)'s GKD trainer to start | [NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html) or [veRL](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html) for scale |
| Evaluating OPD for post-training | [Technical Reports](#technical-reports-and-industrial-recipes) for who is shipping it | [Core OPD Papers](#core-opd-papers) for algorithmic foundations |

> **Key decision:** Do you have access to teacher logits? If yes, start with white-box methods (GKD, Veto, Entropy-Aware OPD). If no, see black-box methods (GAD, OVD) or self-distillation (OPSD, SDFT).

## Start Here

The fastest path to understanding the field:

0. **Survey** — [OPD Survey](https://arxiv.org/abs/2604.00626).
   Comprehensive map of the field: taxonomy, methods, and open problems.
1. **Foundations** — [MiniLLM](https://arxiv.org/abs/2306.08543) and [GKD](https://arxiv.org/abs/2306.13649).
   You will understand the basic student-rollout + teacher-supervision loop.
2. **Practical intuition** — [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/).
   The clearest end-to-end explanation of why and when to use OPD.
3. **Limitations of vanilla OPD** — [Black-Box OPD](https://arxiv.org/abs/2511.10643), [Veto](https://arxiv.org/abs/2601.07155), [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079), [Revisiting OPD](https://arxiv.org/abs/2603.25562).
   You will learn what breaks: instability, diversity collapse, no-logit settings, sampled-token failure modes.
4. **Self-distillation** — [OPSD](https://arxiv.org/abs/2601.18734), [SDFT](https://arxiv.org/abs/2601.19897), [SDPO](https://arxiv.org/abs/2601.20802).
   Drop the external teacher entirely.
5. **Context and experience** — [OPCD](https://arxiv.org/abs/2602.12275) and [OEL](https://arxiv.org/abs/2603.16856).
   Distill prompts, traces, and deployment experience into weights.
6. **2026 efficiency frontier** — [Prefix OPD](https://arxiv.org/abs/2602.15260), [OVD](https://arxiv.org/abs/2601.21968), [PACED](https://arxiv.org/abs/2603.11178), [REOPOLD](https://arxiv.org/abs/2603.11137).
   Cut compute 2x-47x.
7. **Industrial patterns** — [Qwen3](https://arxiv.org/abs/2505.09388), [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780), [GLM-5](https://arxiv.org/abs/2602.15763).
   How labs deploy OPD in production.

## Surveys

- [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626) (2026) — First dedicated OPD survey; organizes methods by feedback signal, teacher access mode, and loss scope.

## Taxonomy

### By Teacher Type

| Teacher Type | Papers |
|---|---|
| External white-box | MiniLLM, GKD, Veto, Entropy-Aware OPD, ExOPD, REOPOLD, PACED, Prefix OPD, Revisiting OPD, Rethinking OPD, Lightning OPD |
| External black-box | Black-Box OPD / GAD, OVD |
| Self-teacher with privileged context | OPSD, SDFT, SDPO, OPSDC, GATES, pi-Distill, RLSD, SDZero |
| Context-conditioned | OPCD, OEL |
| Multiple / lifecycle teachers | MiMo-V2-Flash MOPD, GLM-5, Qwen3, Baichuan-M3, DeepSeek-V4 |

### By Primary Goal

| Goal | Papers |
|---|---|
| Compression / strong-to-weak transfer | MiniLLM, GKD, Qwen3, Prefix OPD, Rethinking OPD, Lightning OPD |
| Post-RL consolidation / skill integration | MiMo MOPD, GLM-5, ExOPD |
| Continual learning | SDFT, OPCD, OEL |
| RL replacement / augmentation | SDPO, RLTF-SD, RLAD, REOPOLD, RLSD, SDZero |
| Reasoning compression | OPSDC |
| Black-box distillation | GAD, OVD |

Many papers span multiple categories. The taxonomy is for orientation, not strict partitioning.

## Core OPD Papers

The ~21 papers that define on-policy distillation for LLMs.

### Foundations

- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) (2023) — Reverse-KL framing for generative LMs; the paper that named the field.
- [GKD: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) (2023) — Unifying formulation spanning on-/off-policy mixtures with flexible divergences.

### Gap-Bridging

- [Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325) (2024) — Interleaved teacher/student sampling to mitigate poor student rollout quality.
- [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643) (2025) — GAD: black-box OPD via discriminator-based reward on student rollouts; no teacher logits needed.

### Stability and Objective Design

- [Veto: Stable On-Policy Distillation through Adaptive Target Reformulation](https://arxiv.org/abs/2601.07155) (2026) — Intermediate target distribution in logit space to stabilize training.
- [Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079) (2026) — Forward-KL on high-entropy teacher tokens to preserve output diversity.
- [ExOPD: Learning beyond Teacher via Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125) (2026) — Casts OPD as dense KL-constrained RL; reward scaling enables teacher-surpassing behavior.
- [REOPOLD: Scaling Reasoning Efficiently via Relaxed On-Policy Distillation](https://arxiv.org/abs/2603.11137) (2026) — Relaxes strict imitation with reward clipping, entropy-based dynamic sampling, and explore-to-refine training.
- [PACED: Distillation at the Frontier of Student Competence](https://arxiv.org/abs/2603.11178) (2026) — Pass-rate weighting focuses learning on the student's competence frontier.
- [Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562) (2026) — Truncated reverse-KL with teacher top-K local support matching; fixes imbalanced signals, unreliable teacher guidance, and tokenizer mismatch in sampled-token OPD.
- [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016) (2026) — Mechanistic analysis of OPD dynamics; identifies compatible thinking patterns and novel teacher capability as success conditions; proposes off-policy cold start and teacher-aligned prompt selection for recovery.
- [The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation](https://arxiv.org/abs/2604.16830) (2026) — Theoretical analysis of OPD-induced overconfidence (information asymmetry, entropy collapse, selection bias); CaOPD replaces confidence targets with student-grounded empirical success rates to decouple capability from calibration.
- [Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models](https://arxiv.org/abs/2604.08527) (2026) — Diagnoses repetition-driven length inflation in iterative OPD; Stable-OPD adds divergence constraints and a rollout-mixture anchor with golden data.

### Self-Distillation

- [OPSD: Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) (2026) — Single model as both teacher and student via privileged information; no external teacher required.
- [SDFT: Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) (2026) — Demonstration-conditioned self-teaching for continual learning with less forgetting.
- [SDPO: Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) (2026) — Converts textual feedback into dense self-teacher signals for RL-like training.
- [Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?](https://arxiv.org/abs/2603.24472) (2026) — Traces self-distillation failures to suppression of epistemic verbalization; task coverage determines whether conciseness helps or hurts.
- [OPSDC: On-Policy Self-Distillation for Reasoning Compression](https://arxiv.org/abs/2603.05433) (2026) — Compresses verbose reasoning using concise privileged self-teachers.
- [GATES: Self-Distillation under Privileged Context with Consensus Gating](https://arxiv.org/abs/2602.20574) (2026) — Consensus-gated asymmetric-context self-distillation without labels or rewards.
- [HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation](https://arxiv.org/abs/2603.23871) (2026) — Privileged self-distillation targeting cliff prompts where RL gradients vanish; provably recovers the KL-regularized optimal policy.
- [RLSD: Self-Distilled RLVR](https://arxiv.org/abs/2604.03128) (2026) — Repurposes self-distillation as token-level credit assignment within GRPO; proves OPSD-style distribution matching under information asymmetry induces irreducible privileged information leakage.
- [SDZero: Self-Revision Turns Binary Rewards into Dense Supervision](https://arxiv.org/abs/2604.12002) (2026) — Generator-reviser dual-role self-distillation requiring only binary rewards; the reviser converts outcome-level feedback into token-level supervision on student rollouts without any external teacher or demonstrations.
- [OPSDL: On-Policy Self-Distillation for Long-Context Language Models](https://arxiv.org/abs/2604.17535) (2026) — Long-context self-distillation: short-context distribution of the same model serves as a co-evolving token-level reverse-KL teacher for student rollouts under long context.

### Context and Experience Internalization

- [OPCD: On-Policy Context Distillation for Language Models](https://arxiv.org/abs/2602.12275) (2026) — Context-conditioned teacher on student rollouts; distills system prompts and experiential knowledge.
- [OEL: Online Experiential Learning for Language Models](https://arxiv.org/abs/2603.16856) (2026) — Deployment loop using OPCD for consolidating interaction traces into weights.
- [Aligning Language Models from User Interactions](https://arxiv.org/abs/2603.12273) (2026) — Hindsight self-distillation from user follow-ups: student rolls out under the original prompt; same model conditioned on the user's follow-up serves as the token-level reverse-KL teacher.

### Efficiency Variants

- [Prefix OPD: Fast and Effective On-policy Distillation from Reasoning Prefixes](https://arxiv.org/abs/2602.15260) (2026) — Distills only reasoning prefixes, cutting training FLOPs 2x-47x.
- [OVD: On-policy Verbal Distillation](https://arxiv.org/abs/2601.21968) (2026) — Trajectory-level verbal scoring instead of token-level logit matching; reduces memory and relaxes alignment requirements.
- [pi-Distill: Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942) (2026) — Training-time privileged information in agentic settings where only actions are observable.
- [Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation](https://arxiv.org/abs/2604.13010) (2026) — Precomputes teacher log-probs once over SFT rollouts to eliminate the live-teacher server; 4x speedup via a teacher-consistency condition.

## Adjacent and Enabling Work

Papers that are not canonical OPD but matter for understanding or deploying it.

### Cross-Tokenizer and Model-Family Enablers

- [ULD: Towards Cross-Tokenizer Distillation](https://arxiv.org/abs/2402.12030) (2024) — Universal Logit Distillation; foundational enabler for cross-family OPD.
- [Multi-Level OT for Universal Cross-Tokenizer KD](https://arxiv.org/abs/2412.14528) (2024) — Token- and sequence-level optimal transport for cross-tokenizer KD.
- [CDM: Enhancing Cross-Tokenizer KD with Contextual Dynamical Mapping](https://arxiv.org/abs/2502.11104) (2025) — Contextual dynamic mapping for vocabulary alignment.
- [Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083) (2025) — Approximate likelihood matching across fundamentally different tokenizers.
- [Cross-Tokenizer Likelihood Scoring Algorithms](https://arxiv.org/abs/2512.14954) (2025) — Exact and approximate sequence likelihood scoring across BPE vocabularies.
- [DSKD: A Dual-Space Framework for General KD](https://arxiv.org/abs/2504.11426) (2025) — Unifies output spaces; supports on- and off-policy KD between any two LLMs.
- [GOLD: Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) (2025) — Cross-tokenizer OPD with TRL integration.
- [CTPD: Cross Tokenizer Preference Distillation](https://arxiv.org/abs/2601.11865) (2026) — Aligned-span projection maps teacher and student tokens to shared character spans; teacher-anchored DPO with cross-tokenizer importance sampling for white-box preference transfer.
- [DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation](https://arxiv.org/abs/2602.21669) (2026) — Dual-space token weighting plus Soft-DTW differentiable sequence alignment for cross-family teacher-student transfer.
- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466) (2026) — Converts teacher distributions to byte-level probabilities; adds a byte-level decoder to the student to enable distillation across mismatched tokenizers.

### Mismatch Mitigation and Student Quality

- [DistiLLM](https://arxiv.org/abs/2402.03898) (2024) — Skew-KL with adaptive off-policy use of student-generated outputs.
- [Exploring and Enhancing Distribution Transfer in KD](https://arxiv.org/abs/2409.12512) (2024) — Analyzes reverse-KL with student-generated output; proposes OKD.
- [FIRST: Efficient Trustworthy Distillation](https://arxiv.org/abs/2408.12168) (2024) — Teacher recalibration for trustworthy offline KD.
- [Multi-Granularity Semantic Revision](https://arxiv.org/abs/2407.10068) (2024) — Sequence correction for low-quality student-generated outputs.
- [Warmup-Distill](https://arxiv.org/abs/2502.11766) (2025) — Bridges distribution mismatch before distillation begins.
- [TAID: Temporally Adaptive Interpolated Distillation](https://arxiv.org/abs/2501.16937) (2025) — Addresses teacher-student mismatch via adaptive interpolation.
- [DistiLLM-2](https://arxiv.org/abs/2503.07067) (2025) — Contrastive extension; student-generated outputs collected per epoch.
- [SpecKD: Speculative Decoding for Effective KD](https://arxiv.org/abs/2510.24021) (2025) — Speculative-decoding-inspired selective token-level losses.
- [Knowledge Distillation with Training Wheels](https://arxiv.org/abs/2502.17717) (2025) — Entropy-regularized value optimization with on-/off-policy demonstrations.
- [Revealing the Power of Post-Training via KD](https://arxiv.org/abs/2509.26497) (2025) — Offline on-policy KD: student generates, then teacher labels.
- [TSD-KD: Explain in Your Own Words](https://arxiv.org/abs/2603.13260) (2026) — Student proposes candidates, teacher reranks, selective token distillation.
- [SSD: Embarrassingly Simple Self-Distillation Improves Code Generation](https://arxiv.org/abs/2604.01193) (2026) — Temperature-shifted self-sampling plus SFT with no teacher or verifier; identifies precision-exploration conflict in token distributions.
- [AdaSwitch: Balancing Exploration and Guidance in KD via Adaptive Switching](https://arxiv.org/abs/2510.07842) (2025) — Adaptively switches between on-policy student rollouts and off-policy teacher data using a context-aware divergence threshold.
- [DDT: Towards On-Policy SFT via Distribution Discriminant Theory](https://arxiv.org/abs/2602.12222) (2026) — In-Distribution Finetuning and Hinted Decoding realign training data to the student's evolving distribution; matches offline RL at SFT cost.
- [DASD: Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning](https://arxiv.org/abs/2601.09088) (2026) — On-policy correction pipeline addressing distribution mismatch, capacity misalignment, and exposure bias in sequence-level CoT distillation.
- [SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting](https://arxiv.org/abs/2604.10688) (2026) — Routes correct rollouts to student-perplexity-weighted MLE and incorrect ones to teacher-perplexity-weighted KL distillation; per-prompt group-level normalization.
- [TIP: Token Importance in On-Policy Distillation](https://arxiv.org/abs/2604.14084) (2026) — Selective token training on high-entropy positions and confidently-wrong low-entropy positions; matches full-token baselines at lower memory.
- [DP-OPD: Differentially Private On-Policy Distillation for Language Models](https://arxiv.org/abs/2604.04461) (2026) — Standard student-rollout + frozen-teacher OPD with DP-SGD on student updates; first OPD recipe compatible with sample-level differential privacy.
- [Distillation Traps and Guards: A Calibration Knob for LLM Distillability](https://arxiv.org/abs/2604.18963) (2026) — Diagnoses tail noise, off-policy instability, and teacher-student gap as KD/OPD failure modes; post-hoc calibrates teachers via RFT to control distillability.

### Preference, Reward-Guided, and Hybrid RL+KD

- [Direct Preference Knowledge Distillation](https://arxiv.org/abs/2406.19774) (2024) — Preference-aware KD combining reverse-KL with implicit reward objectives.
- [Online Knowledge Distillation with Reward Guidance](https://arxiv.org/abs/2505.18952) (2025) — Sequential KD via preference optimization; offline and online variants.
- [KDRL](https://arxiv.org/abs/2506.02208) (2025) — Unified reverse-KL KD with RL in a single post-training objective.
- [RLTF-SD: Expanding RL via Text Feedback](https://arxiv.org/abs/2602.02482) (2026) — Internalizes text feedback via self-distillation.
- [RLAD: Reinforcement-aware KD for LLM Reasoning](https://arxiv.org/abs/2602.22495) (2026) — Trust-region ratio distillation on student rollouts.
- [Multi-Token Prediction via Self-Distillation](https://arxiv.org/abs/2602.06019) (2026) — Online self-distillation for multi-token prediction and faster inference.
- [ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](https://arxiv.org/abs/2509.25100) (2025) — Mixed-policy teacher/student preference distillation using student-generated outputs; enables black-box cross-architecture transfer.
- [SRPO: Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing](https://arxiv.org/abs/2604.02288) (2026) — Routes correct student rollouts to reward-based RL and failed ones to self-distillation; unifies GRPO and SDPO.
- [KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation](https://arxiv.org/abs/2504.19024) (2025) — K-step return via Bellman equation replaces high-variance single-step REINFORCE in sequence-level OPD.
- [Rethinking LLM Distillation: A Constrained MDP Perspective](https://arxiv.org/abs/2509.22921) (2025) — Maximizes task reward subject to hard KL constraint against the teacher; avoids manual Lagrangian tuning.
- [RLKD: Distilling LLMs' Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.16142) (2025) — Generative Structure Reward Model captures multi-branch reasoning structure on student rollouts; outperforms SFT-RL pipelines on 0.1% data.
- [LUFFY: Learning to Reason under Off-Policy Guidance](https://arxiv.org/abs/2504.14945) (2025) — Mixed-policy GRPO combining on-policy rollouts with off-policy teacher traces via regularized importance sampling.
- [BOND: Aligning LLMs with Best-of-N Distillation](https://arxiv.org/abs/2407.14622) (2024) — RL that mimics best-of-N sampling via Jeffreys-divergence distribution matching; eliminates inference-time BoN cost.
- [Faster WIND: Accelerating Iterative Best-of-N Distillation for LLM Alignment](https://arxiv.org/abs/2410.20727) (2024) — Game-theoretic framing of iterative BoN as self-play; win-rate dominance optimization with sample-efficiency guarantees (AISTATS 2025).
- [AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](https://arxiv.org/abs/2503.02832) (2025) — Casts RLHF as token-level distillation by injecting DPO rewards; contrastive token-adaptive optimization (ACL 2025).
- [KEPO: Knowledge-Enhanced Preference Optimization for Reinforcement Learning with Reasoning](https://arxiv.org/abs/2602.00400) (2026) — Quality-gated on-policy distillation on high-quality trajectories combined with knowledge-enhanced exploration via teacher hints.
- [𝒳-KD: General Experiential Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2602.12674) (2026) — Jointly models the teacher's reward and performs policy distillation so the student learns inside the teacher's original learning environment.
- [ExGRPO: Probing to Refine — Reinforcement Distillation of LLMs via Explanatory Inversion](https://arxiv.org/abs/2603.19266) (2026) — Explanatory probes plus dialogue-structure utility bonus reward coherent reasoning over memorized answers.
- [HPD: Hybrid Policy Distillation for LLMs](https://arxiv.org/abs/2604.20244) (2026) — Unified reweighted-log-likelihood framework combining forward/reverse KL with off-policy and lightweight on-policy sampling.
- [NPO: Near-Future Policy Optimization](https://arxiv.org/abs/2604.20733) (2026) — Uses a later checkpoint of the same policy as the teacher; AutoNPO adaptively triggers the switch to maximize learning signal in RLVR.

### Self-Play and Iterative Bootstrapping

- [SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) (2024) — Self-play loop where the model distinguishes its own generations from human references; ICML 2024.
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) (2024) — Iterative DPO with the model as LLM-as-Judge providing self-rewards on its own generations; supervision-equivalent feedback from same-model evaluation.
- [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://arxiv.org/abs/2501.04519) (2025) — MCTS-guided self-evolution: policy and process reward model co-improve from scratch via code-augmented reasoning.
- [rStar2-Agent: Agentic Reasoning Technical Report](https://arxiv.org/abs/2508.20722) (2025) — GRPO with Resample-on-Correct rollouts plus multi-stage SFT→RL recipe for a 14B agentic reasoner.
- [π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data](https://arxiv.org/abs/2604.14054) (2026) — Examiner generates tasks plus question-construction-paths (QCPs); same-scale teacher uses QCPs as privileged context to densely supervise student rollouts via reverse KL; turns sparse-reward self-play into dense self-distillation.

### Agent Distillation, Multimodal, and Other Extensions

- [Structured Agent Distillation](https://arxiv.org/abs/2505.13820) (2025) — Queries teacher online to avoid distribution drift in agent settings.
- [From Deferral to Learning: Online In-Context KD for LLM Cascades](https://arxiv.org/abs/2509.22984) (2025) — Teacher-student cascade with reusable online knowledge store.
- [AllMem](https://arxiv.org/abs/2602.13680) (2026) — Offline on-policy distillation for long-context modeling.
- [Video-OPD](https://arxiv.org/abs/2602.02994) (2026) — OPD for temporal video grounding in multimodal LLMs.
- [Reinforced Attention Learning](https://arxiv.org/abs/2602.04884) (2026) — On-policy attention distillation for multimodal models.
- [SCoRe: From Correction to Mastery via Reinforced Distillation of LLM Agents](https://arxiv.org/abs/2509.14257) (2025) — Student generates agent trajectories; teacher intervenes at first critical error for on-policy corrective distillation.
- [VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation](https://arxiv.org/abs/2510.23497) (2025) — Text-only teacher distills reasoning into VLM student via student-generated traces with combined GRPO and OPD.
- [X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs](https://arxiv.org/abs/2603.24596) (2026) — Student on-policy rollouts with token-level teacher feedback for cross-modal speech-LLM distillation.
- [VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation](https://arxiv.org/abs/2603.26666) (2026) — Reverse-KL on-policy distillation bridging offline SFT and online RL for robotic manipulation.
- [TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents](https://arxiv.org/abs/2604.24005) (2026) — Short-to-long trajectory-depth curriculum that mitigates trajectory-level KL instability when accumulated multi-turn errors push student behavior beyond teacher support.
- [LLM4Teach: Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents](https://arxiv.org/abs/2311.13373) (2023) — LLM teacher distills into a small RL agent that surpasses the teacher through environment interaction.
- [RPD: Refined Policy Distillation — From VLA Generalists to RL Experts](https://arxiv.org/abs/2503.05833) (2025) — Teacher VLA actions guide the student policy during RL exploration; combines RL with behavioral cloning (IROS 2026).
- [π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](https://arxiv.org/abs/2510.14974) (2025) — Imitation distillation aligns student flow-model policy trajectories with teacher trajectories under standard flow matching loss (ICLR 2026).
- [Step-Audio-R1 Technical Report](https://arxiv.org/abs/2511.15848) (2025) — Modality-Grounded Reasoning Distillation produces audio reasoning chains grounded in acoustic features rather than hallucinated text.
- [OPD-AVMP: On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning](https://arxiv.org/abs/2604.07944) (2026) — Generalized on-policy KD for LLM-based driving planners; 5x compression at near-teacher performance.
- [CORD: Bridging the Audio–Text Reasoning Gap via Weighted On-policy Cross-modal Distillation](https://arxiv.org/abs/2601.16547) (2026) — Audio-conditioned student rollouts; text-conditioned same model as in-model teacher; importance-weighted reverse KL on early/critical tokens plus GRPO at sequence level.
- [Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents](https://arxiv.org/abs/2604.10674) (2026) — Student rolls out under plain prompt; same model under skill-augmented prompt serves as token-level self-teacher for multi-turn agent training.

### Speculative Decoding (Draft-Model Training)

Draft-model training for speculative decoding shares OPD's core loop: the draft (student) generates, the target (teacher) verifies, and the draft is updated to match. Included for breadth even though the goal is inference acceleration rather than student capability.

- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177) (2023) — Continuously updates the draft model on observed user queries via knowledge distillation; 1.42x-2.17x latency gains.
- [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461) (2023) — Aligns draft with target via on-policy data and task-tailored divergence; foundational draft-model OPD recipe (ICLR 2024).
- [HASS: Learning Harmonized Representations for Speculative Sampling](https://arxiv.org/abs/2408.15766) (2024) — Harmonized objective and context distillation to fix train-decoding inconsistency in speculative sampling.
- [Falcon: Faster and Parallel Inference of Large Language Models through Enhanced Semi-Autoregressive Drafting](https://arxiv.org/abs/2412.12639) (2024) — Coupled Sequential Glancing Distillation strengthens inter-token dependencies in semi-autoregressive drafters.
- [CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter](https://arxiv.org/abs/2502.16880) (2025) — Cross-step representation alignment for multi-step drafter training (ACL 2025).
- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) (2025) — Direct token prediction with multi-layer feature fusion under on-policy training-time test; up to 6.5x speedup.
- [MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models](https://arxiv.org/abs/2505.10526) (2025) — Adapts an SLM into a VLM drafter via self-distilled visual instruction tuning.
- [DVI: Draft, Verify, and Improve — Toward Training-Aware Speculative Decoding](https://arxiv.org/abs/2510.05421) (2025) — Self-speculative drafter trained online from verifier decisions via KL→RL schedule with reward-masked cross-entropy.
- [ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems](https://arxiv.org/abs/2510.26475) (2025) — Evolves the drafter during RL via reward-weighted distillation on rollouts.
- [DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution](https://openreview.net/forum?id=CRgWv0kWjF) (2026) — Multimodal speculative-reasoning drafter trained for faithfulness to target trajectories with verifier-gated parallel execution.

### Precursors

- [Autoregressive KD through Imitation Learning](https://arxiv.org/abs/2009.07253) (2020) — Early precursor framing sequence-model KD as imitation learning.
- [Learning by Distilling Context](https://arxiv.org/abs/2209.15189) (2022) — Context distillation; key precursor to OPCD and OEL.

## Technical Reports and Industrial Recipes

Production training pipelines that use OPD as a post-training stage.

| Year | System | OPD Usage | Link |
|------|--------|-----------|------|
| 2024 | Gemma 2 | KD as alternative to next-token prediction for the 2B and 9B student models | [paper](https://arxiv.org/abs/2408.00118) |
| 2025 | Qwen3 | Strong-to-weak; off-policy then on-policy distillation | [paper](https://arxiv.org/abs/2505.09388) |
| 2025 | Qwen3-Omni | Off-policy then on-policy distillation before GSPO | [paper](https://arxiv.org/abs/2509.17765) |
| 2025 | GLM-4.5 / 4.6 | Multi-stage post-training with expert model iteration and RL (precursor to GLM-5's explicit OPD) | [paper](https://arxiv.org/abs/2508.06471) |
| 2025 | HY-MT1.5 | Multi-stage translation: SFT + OPD + RL | [paper](https://arxiv.org/abs/2512.24092) |
| 2026 | MiMo-V2-Flash | Multi-Teacher OPD (MOPD) as post-training stage | [paper](https://arxiv.org/abs/2601.02780) |
| 2026 | GLM-5 | On-policy cross-stage distillation to recover earlier skills | [paper](https://arxiv.org/abs/2602.15763) |
| 2026 | Typhoon-S | Minimal sovereign recipe: SFT + OPD + small-scale RFT | [paper](https://arxiv.org/abs/2601.18129) |
| 2026 | Nemotron-Cascade 2 | Cascade RL + multi-domain on-policy distillation | [paper](https://arxiv.org/abs/2603.19220) |
| 2026 | Baichuan-M3 | Three-stage: task RL, offline policy distillation, multi-teacher OPD | [paper](https://arxiv.org/abs/2602.06570) |
| 2026 | MobileLLM-R1.5 | Final-stage on-policy KD as primary improvement over R1 | model card |
| 2026 | Nanbeige4-3B-Thinking | OPD preferred over off-policy for math reasoning | [model card](https://huggingface.co/Nanbeige/Nanbeige4-3B-Thinking-2510) |
| 2026 | DeepSeek-V4 | Two-stage post-training: domain-expert SFT+GRPO, then unified model consolidation via on-policy distillation | [report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |
| 2026 | Qwen3.5-Omni | Specialist teacher distillation, then privileged-input self-distillation aligning audio-conditioned outputs to text-conditioned responses (labeled OPD by the report) | [paper](https://arxiv.org/abs/2604.15804) |
| 2026 | HY-Embodied-0.5 | Large-to-small on-policy distillation transfers 32B embodied-reasoning behavior into the 2B edge variant: student rollouts, teacher token-level supervision at the same prefixes | [paper](https://arxiv.org/abs/2604.07430) |

## Frameworks, Tools, and Implementations

### Training Frameworks

| Framework | Description | Link |
|---|---|---|
| TRL | GKD, GOLD, and MiniLLM trainers; most accessible starting point | [docs](https://huggingface.co/docs/trl) |
| NeMo-RL | Multi-teacher and cross-tokenizer OPD at scale | [docs](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html), [repo](https://github.com/NVIDIA-NeMo/RL) |
| veRL | Async on-policy KD trading strict on-policy guarantees for throughput | [docs](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html) |
| MS-Swift | GKD and OPSD sections in the ModelScope ecosystem | [docs](https://swift.readthedocs.io/en/latest/) |
| EasyDistill | Comprehensive KD toolkit for black-box and white-box LLM distillation | [paper](https://arxiv.org/abs/2505.20888) |
| KDFlow | Off-policy, on-policy, and cross-tokenizer distillation via decoupled backends | [paper](https://arxiv.org/abs/2603.01875) |
| slime | Unified RL stack supporting on-policy distillation and hindsight hints | [repo](https://github.com/) |
| OpenClaw-RL | Agentic RL stack with hindsight-guided OPD | [paper](https://arxiv.org/abs/2603.10165) |
| NexRL | Dedicated on-policy distillation recipes | [repo](https://github.com/nex-agi/NexRL) |
| SkyRL | OPD examples and blog resources | [repo](https://github.com/NovaSky-AI/SkyRL) |
| ATLAS | Continual-learning framework using GKD/GRPO from runtime traces | [docs](https://docs.arc.computer/introduction) |
| AReaL | OPD and KDRL implementation over student-sampled trajectories with teacher log-prob guidance | [docs](https://github.com/inclusionAI/AReaL/blob/main/docs/en/algorithms/distillation.md) |
| SpecForge | Open-source training framework for speculative draft models with EAGLE-3 support, target-draft decoupling, and hybrid parallelism | [paper](https://arxiv.org/abs/2603.18567), [repo](https://github.com/sgl-project/SpecForge) |

### Code, Tutorials, and Guides

- [OPSD](https://github.com/siyan-zhao/OPSD) — Official code for Self-Distilled Reasoner / OPSD.
- [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) — Best single-article introduction. Covers concepts, intuition, and practical use cases.
- [Unlocking On-Policy Distillation for Any Model Family (GOLD)](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) — Cross-tokenizer OPD walkthrough with TRL code.
- [SCOPE](https://github.com/machine981/SCOPE) — Official implementation of SCOPE's dual-path OPD: student-PPL-weighted MLE for correct rollouts, teacher-PPL-weighted KL for incorrect.
- [CaOPD](https://github.com/SalesforceAIResearch/CaOPD) — Official implementation of CaOPD: K student rollouts → empirical success rate → confidence target replacement → standard reverse-KL OPD.
- [OPSD-OnPolicyDistillation](https://github.com/HJSang/OPSD_OnPolicyDistillation) — verl-based OPD implementation with separate-teacher distillation, agent-loop rollouts, and memory-efficient teacher/student execution.

## Acknowledgments

This list draws on the parallel curation effort at [thinkwee/AwesomeOPD](https://github.com/thinkwee/AwesomeOPD), which provided pointers to several papers (notably the speculative-decoding draft-model training, BoN distillation, self-play, and additional industrial reports) that broadened the scope of this list. The two lists are organized differently — thinkwee/AwesomeOPD groups by feedback signal and access mode; this list groups by methodological role — and are best read together.

## Contributing

Contributions welcome! Please open a PR if you know of papers, reports, or tools related to on-policy distillation. See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed criteria, section placement, and formatting guidelines.

- **Inclusion criteria:** The work should involve student rollouts as central to the learning signal, or directly enable OPD deployment (cross-tokenizer, frameworks, etc.).
- **Entry format:** `[Title](url) (Year) — One-line description.`

## Citation

If you find this resource useful, please cite it as:

```bibtex
@software{awesome-on-policy-distillation,
  title = {{Awesome On-Policy Distillation}},
  author = {Liu, Chris Yuhao},
  year = {2026},
  doi = {10.5281/zenodo.19411493},
  url = {https://github.com/chrisliu298/awesome-on-policy-distillation},
  version = {v1.0.0}
}
```

---

*Last updated: 2026-04-28. Coverage: core OPD papers, adjacent work (including speculative-decoding draft-model training and self-play / iterative bootstrapping), surveys, technical reports, and tooling through April 2026.*
