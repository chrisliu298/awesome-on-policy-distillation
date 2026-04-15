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
  - [Agent Distillation, Multimodal, and Other Extensions](#agent-distillation-multimodal-and-other-extensions)
  - [Precursors](#precursors)
- [Technical Reports and Industrial Recipes](#technical-reports-and-industrial-recipes)
- [Frameworks, Tools, and Implementations](#frameworks-tools-and-implementations)
  - [Training Frameworks](#training-frameworks)
  - [Code, Tutorials, and Guides](#code-tutorials-and-guides)
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
| External white-box | MiniLLM, GKD, Veto, Entropy-Aware OPD, ExOPD, REOPOLD, PACED, Prefix OPD, Revisiting OPD, Rethinking OPD |
| External black-box | Black-Box OPD / GAD, OVD |
| Self-teacher with privileged context | OPSD, SDFT, SDPO, OPSDC, GATES, pi-Distill, RLSD |
| Context-conditioned | OPCD, OEL |
| Multiple / lifecycle teachers | MiMo-V2-Flash MOPD, GLM-5, Qwen3, Baichuan-M3 |

### By Primary Goal

| Goal | Papers |
|---|---|
| Compression / strong-to-weak transfer | MiniLLM, GKD, Qwen3, Prefix OPD, Rethinking OPD |
| Post-RL consolidation / skill integration | MiMo MOPD, GLM-5, ExOPD |
| Continual learning | SDFT, OPCD, OEL |
| RL replacement / augmentation | SDPO, RLTF-SD, RLAD, REOPOLD, RLSD |
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

### Self-Distillation

- [OPSD: Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) (2026) — Single model as both teacher and student via privileged information; no external teacher required.
- [SDFT: Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) (2026) — Demonstration-conditioned self-teaching for continual learning with less forgetting.
- [SDPO: Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) (2026) — Converts textual feedback into dense self-teacher signals for RL-like training.
- [Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?](https://arxiv.org/abs/2603.24472) (2026) — Traces self-distillation failures to suppression of epistemic verbalization; task coverage determines whether conciseness helps or hurts.
- [OPSDC: On-Policy Self-Distillation for Reasoning Compression](https://arxiv.org/abs/2603.05433) (2026) — Compresses verbose reasoning using concise privileged self-teachers.
- [GATES: Self-Distillation under Privileged Context with Consensus Gating](https://arxiv.org/abs/2602.20574) (2026) — Consensus-gated asymmetric-context self-distillation without labels or rewards.
- [HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation](https://arxiv.org/abs/2603.23871) (2026) — Privileged self-distillation targeting cliff prompts where RL gradients vanish; provably recovers the KL-regularized optimal policy.
- [RLSD: Self-Distilled RLVR](https://arxiv.org/abs/2604.03128) (2026) — Repurposes self-distillation as token-level credit assignment within GRPO; proves OPSD-style distribution matching under information asymmetry induces irreducible privileged information leakage.

### Context and Experience Internalization

- [OPCD: On-Policy Context Distillation for Language Models](https://arxiv.org/abs/2602.12275) (2026) — Context-conditioned teacher on student rollouts; distills system prompts and experiential knowledge.
- [OEL: Online Experiential Learning for Language Models](https://arxiv.org/abs/2603.16856) (2026) — Deployment loop using OPCD for consolidating interaction traces into weights.

### Efficiency Variants

- [Prefix OPD: Fast and Effective On-policy Distillation from Reasoning Prefixes](https://arxiv.org/abs/2602.15260) (2026) — Distills only reasoning prefixes, cutting training FLOPs 2x-47x.
- [OVD: On-policy Verbal Distillation](https://arxiv.org/abs/2601.21968) (2026) — Trajectory-level verbal scoring instead of token-level logit matching; reduces memory and relaxes alignment requirements.
- [pi-Distill: Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942) (2026) — Training-time privileged information in agentic settings where only actions are observable.

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

### Precursors

- [Autoregressive KD through Imitation Learning](https://arxiv.org/abs/2009.07253) (2020) — Early precursor framing sequence-model KD as imitation learning.
- [Learning by Distilling Context](https://arxiv.org/abs/2209.15189) (2022) — Context distillation; key precursor to OPCD and OEL.

## Technical Reports and Industrial Recipes

Production training pipelines that use OPD as a post-training stage.

| Year | System | OPD Usage | Link |
|------|--------|-----------|------|
| 2025 | Qwen3 | Strong-to-weak; off-policy then on-policy distillation | [paper](https://arxiv.org/abs/2505.09388) |
| 2025 | Qwen3-Omni | Off-policy then on-policy distillation before GSPO | [paper](https://arxiv.org/abs/2509.17765) |
| 2025 | HY-MT1.5 | Multi-stage translation: SFT + OPD + RL | [paper](https://arxiv.org/abs/2512.24092) |
| 2026 | MiMo-V2-Flash | Multi-Teacher OPD (MOPD) as post-training stage | [paper](https://arxiv.org/abs/2601.02780) |
| 2026 | GLM-5 | On-policy cross-stage distillation to recover earlier skills | [paper](https://arxiv.org/abs/2602.15763) |
| 2026 | Typhoon-S | Minimal sovereign recipe: SFT + OPD + small-scale RFT | [paper](https://arxiv.org/abs/2601.18129) |
| 2026 | Nemotron-Cascade 2 | Cascade RL + multi-domain on-policy distillation | [paper](https://arxiv.org/abs/2603.19220) |
| 2026 | Baichuan-M3 | Three-stage: task RL, offline policy distillation, multi-teacher OPD | [paper](https://arxiv.org/abs/2602.06570) |
| 2026 | MobileLLM-R1.5 | Final-stage on-policy KD as primary improvement over R1 | model card |
| 2026 | Nanbeige4-3B-Thinking | OPD preferred over off-policy for math reasoning | [model card](https://huggingface.co/Nanbeige/Nanbeige4-3B-Thinking-2510) |

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

### Code, Tutorials, and Guides

- [OPSD](https://github.com/siyan-zhao/OPSD) — Official code for Self-Distilled Reasoner / OPSD.
- [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) — Best single-article introduction. Covers concepts, intuition, and practical use cases.
- [Unlocking On-Policy Distillation for Any Model Family (GOLD)](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) — Cross-tokenizer OPD walkthrough with TRL code.

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

*Last updated: 2026-04-14. Coverage: core OPD papers, adjacent work, surveys, technical reports, and tooling through April 2026.*
