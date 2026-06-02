# ⚗️ Awesome On-Policy Distillation

<p align="center">
  <a href="https://awesome.re"><img src="https://img.shields.io/badge/Awesome-%E2%9A%97%EF%B8%8F_On--Policy_Distillation-000000?style=for-the-badge&labelColor=000000" alt="Awesome On-Policy Distillation"></a>
</p>

<p align="center">
  <!-- entry-count-start --><a href="#contents"><img src="https://img.shields.io/badge/Entries-287-000000?style=for-the-badge&labelColor=000000" alt="Entries"></a><!-- entry-count-end -->
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/stargazers"><img src="https://img.shields.io/github/stars/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Stars&labelColor=000000&color=000000" alt="GitHub Stars"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/network/members"><img src="https://img.shields.io/github/forks/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Forks&labelColor=000000&color=000000" alt="GitHub Forks"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/commits"><img src="https://img.shields.io/github/last-commit/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Last%20Commit&labelColor=000000&color=000000" alt="Last Commit"></a>
</p>

A curated collection of papers, technical reports, frameworks, and tools for on-policy distillation (OPD) of large language models.

> **On-policy distillation** trains a student on samples from its own evolving policy, while a teacher (external, privileged, or self-conditioned) provides dense supervision on those same samples.

On-policy distillation (OPD) trains a student on trajectories sampled from its own policy while a teacher scores the student-visited prefixes with dense token-level guidance. This on-policy data collection reduces the train-inference distribution gap that affects off-policy KD/SFT on fixed traces. Depending on the estimator, OPD looks like GKD on student rollouts or policy-gradient/RL with teacher-defined per-token KL/log-prob rewards, making the natural contrast sparse outcome-reward RL rather than RL as a whole. As of 2026, OPD is a standard post-training primitive at Alibaba (Qwen3), DeepSeek (V4), Xiaomi (MiMo), Zhipu (GLM-5), NVIDIA (Nemotron-Cascade 2), and others.

**Shipping today?** Jump to [Frameworks and Implementations](#frameworks-and-implementations). **New to OPD?** Read [Start Here](#start-here).

## Contents

- [Start Here](#start-here)
- [Surveys and Essays](#surveys-and-essays)
- [Core OPD Papers](#core-opd-papers)
  - [Foundations](#foundations)
  - [Gap-Bridging](#gap-bridging)
  - [Stability and Objective Design](#stability-and-objective-design)
  - [Self-Distillation](#self-distillation)
  - [Context and Experience Internalization](#context-and-experience-internalization)
  - [Efficiency, Systems, and Privacy Variants](#efficiency-systems-and-privacy-variants)
- [Taxonomy](#taxonomy)
- [Adjacent and Enabling Work](#adjacent-and-enabling-work)
- [Domain Extensions](#domain-extensions)
- [Technical Reports and Industrial Recipes](#technical-reports-and-industrial-recipes)
- [Frameworks and Implementations](#frameworks-and-implementations)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [Citation](#citation)

## Start Here

A fast path through the field:

1. **Survey.** [OPD Survey](https://arxiv.org/abs/2604.00626) — taxonomy, methods, and open problems in one place.
2. **Foundations.** [MiniLLM](https://arxiv.org/abs/2306.08543), [GKD](https://arxiv.org/abs/2306.13649), and [ExOPD](https://arxiv.org/abs/2602.12125) — the core student-rollout plus teacher-supervision loop, including its dense KL-constrained RL framing.
3. **Practical intuition.** [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/) — the clearest end-to-end explanation of why and when OPD applies.
4. **When OPD works and when it breaks.** [Revisiting OPD](https://arxiv.org/abs/2603.25562), [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079), and [Rethinking OPD](https://arxiv.org/abs/2604.13016) — failure modes (instability, diversity collapse, tokenizer mismatch) and success conditions (compatible thinking patterns, novel teacher capability).
5. **No teacher logits.** [Black-Box OPD](https://arxiv.org/abs/2511.10643) — discriminator-based reward when the teacher is API-only.
6. **No teacher at all.** [OPSD](https://arxiv.org/abs/2601.18734) and [SDFT](https://arxiv.org/abs/2601.19897) — same model as student and self-teacher.
7. **Context and experience.** [OPCD](https://arxiv.org/abs/2602.12275) and [OEL](https://arxiv.org/abs/2603.16856) — distill prompts and deployment traces into weights.
8. **Industrial recipes.** [Qwen3](https://arxiv.org/abs/2505.09388), [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf), [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780), [GLM-5](https://arxiv.org/abs/2602.15763) — how labs ship OPD in production.

**Key decision:** access to teacher logits? Yes → white-box (GKD, Veto, Entropy-Aware OPD). No → black-box (GAD, OVD) or self-distillation (OPSD, SDFT).

## Surveys and Essays

### Surveys and Position Papers

- [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626) *(2026)* — First dedicated OPD survey; organizes methods by feedback signal, teacher access mode, and loss scope.
- [A Brief Overview: On-Policy Self-Distillation in Large Language Models](https://arxiv.org/abs/2605.18141) *(2026)* — Beginner-oriented overview of on-policy self-distillation, cataloguing privileged-context designs where a single model is both teacher and student.
- [Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation](https://arxiv.org/abs/2605.22731) *(2026)* — Reframes SFT/RL/OPD by training-state source rather than loss, explaining why OPD's student-sampled states beat a degraded teacher.

### Essays, Blog Posts, and Walkthroughs

- [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) *(2025)* — Best single-article introduction. Covers concepts, intuition, and practical use cases.
- [Unlocking On-Policy Distillation for Any Model Family (GOLD)](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) *(2025)* — Cross-tokenizer OPD walkthrough with TRL code.
- [Distilling 100B+ Models 40x Faster with TRL](https://huggingface.co/spaces/HuggingFaceTB/trl-distillation-trainer) *(2026)* — HF engineering walkthrough of TRL's `DistillationTrainer` scaling tricks; ~40× speedup, validated on Qwen3-235B → Qwen3-4B math.
- [Multi-Teacher On-Policy Distillation: A New Post-Training Primitive](https://yumoxu.notion.site/multi-teacher-on-policy-distillation) *(2026)* — Yumo Xu surveys MOPD as a post-training primitive across MiMo-V2-Flash, GLM-5, Nemotron-Cascade 2, DeepSeek-V4.
- [On-Policy Distillation: Theory & Practice in Model Merging](https://www.notion.so/On-Policy-Distillation-Theory-Practice-in-Model-Merging-2f44795a3e8b801cbedee2c96a23c788) *(2026)* — ByteDance Seed framing OPD as entropy-regularized RL; cross-tokenizer pitfalls and reward hacking in agent merging.
- [On SFT, RL, and on-policy distillation](https://x.com/willccbb/status/2050038277454143918) *(2026)* — Will Brown's essay on OPD via SFT-vs-RL compounding and gradient geometry; pointers toward an optimal teacher.
- [SFT, RL, and OPD Through a Distributional Lens](https://x.com/nrehiew_/status/2053482349300797526) *(2026)* — wh's distributional-geometry framing; experiment shows OPD students from SFT and RL teachers converge and forget less.
- [On Policy Self Distillation](https://x.com/ar0cket1/status/2054108160450064571) *(2026)* — KL-geometry study showing OPSD inverts OPD's per-token sign and suffers larger KL shocks that GEPA hint evolution roughly halves.
- [What Apple found out about On-Policy Distillation](https://x.com/neural_avb/status/2054585001757614172) *(2026)* — AVB's tutorial-style breakdown of "Unmasking OPD"; training-free gradient-alignment for predicting student-teacher fit.
- [OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践 / OPD Deep Dive: From Mathematical Derivation to DeepSeek V4, SWIFT, and verl Practice](https://zhuanlan.zhihu.com/p/2033212181823608430) *(2026)* — Chinese-language Zhihu deep-dive deriving OPD's sequence- and token-level reverse-KL; maps variants to MiniLLM, GKD, verl, DeepSeek V4.
- [重温 On-Policy Distillation / Revisiting On-Policy Distillation](https://yph22.github.io/files/notes/OPD.pdf) *(2026)* — Chinese-language notes deriving OPD as both a SeqKD student-rollout mirror and RL with token-level teacher supervision.
- [The Imitation Game: State of Policy Distillation in Language Model training](https://hackmd.io/@l_WDq7lkQq29Pz-KD1JPNA/r1oNsX9Jfl) *(2026)* — Long-form OPD/OPSD survey with a four-axis failure-modes taxonomy; argues hybrid OPSD and cross-tokenizer OPD as the highest-leverage open problems.

## Core OPD Papers

The papers that define on-policy distillation for LLMs.

**Scope rule:** A paper belongs here if its primary contribution is a new component of the OPD training loop itself — an objective, divergence formulation, stability fix, teacher access-mode variant, self-distillation variant, context-internalization mechanism, or systems/efficiency/privacy constraint applied to that loop — with student rollouts central to the learning signal, evaluated on LLM text generation or reasoning. **Operational test:** if removing the OPD-loop component leaves a working contribution (a working RL recipe, preference method, or KD baseline), the OPD piece is auxiliary → Adjacent. Papers that *enable* OPD (cross-tokenizer alignment, calibration), *compose with* OPD as one component of a larger RL/preference structure, or apply OPD to non-text-reasoning substrates live in [Adjacent and Enabling Work](#adjacent-and-enabling-work) or [Domain Extensions](#domain-extensions).

### Foundations

- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) *(2023)* — Reverse-KL framing for generative LMs; the paper that named the field.
- [GKD: On-Policy Distillation of Language Models — Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) *(2023)* — Unifying formulation spanning on-/off-policy mixtures with flexible divergences.

### Gap-Bridging

- [Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325) *(2024)* — Interleaved teacher/student sampling mitigates poor student rollout quality.
- [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643) *(2025)* — GAD: discriminator-based reward on student rollouts; no teacher logits required.
- [SOD: Step-wise On-policy Distillation for Small Language Model Agents](https://arxiv.org/abs/2605.07725) *(2026)* — Reweights teacher guidance by step-level divergence to avoid tool-induced cascade drift.
- [MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate](https://arxiv.org/abs/2605.01347) *(2026)* — Multi-agent debate consensus as the OPD teacher; extends to agentic tasks via step-level sampling.
- [ROPD: Rubric-based On-policy Distillation](https://arxiv.org/abs/2605.07396) *(2026)* — Black-box OPD using prompt-specific rubrics distilled from teacher-student contrasts to score rollouts.
- [Backtracking When It Strays: Mitigating Dual Exposure Biases in LLM Reasoning Distillation](https://arxiv.org/abs/2605.19433) *(2026)* — Backtracks straying student rollouts to the last safe state for teacher correction, targeting the reversed exposure bias on-policy distillation introduces. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/wangbing1416/MOTAB)
- [Counteraction-Aware Multi-Teacher On-Policy Distillation for General Capability Recovery with Domain Preservation](https://arxiv.org/abs/2605.27115) *(2026)* — Counteraction-aware multi-teacher OPD that decouples conflicting recovery and preservation gradients, recovering general capability from proxy prompts without teacher-aligned prompt coverage.
- [Trust-Region Behavior Blending for On-Policy Distillation](https://arxiv.org/abs/2605.31159) *(2026)* — Warmup samples early prefixes from a teacher-blended behavior policy within a student-centered KL trust region, annealed to zero by warmup's end.
- [Bridging Reasoning Trajectories in On-Policy Distillation via Near-Future Guidance](https://arxiv.org/abs/2606.00305) *(2026)* — Spreads teacher guidance across a near-future token window, using trajectory drift to find true reasoning forks rather than high-loss single tokens.
- [Trust Region On-Policy Distillation](https://arxiv.org/abs/2606.01249) *(2026)* — Restricts reverse-KL distillation to teacher-reliable trust regions on student rollouts, applying forward-KL to mismatched outlier tokens instead.
- [OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification](https://arxiv.org/abs/2606.01476) *(2026)* — Replaces teacher logits with chunk-level semantic verification from Monte Carlo rollouts, enabling on-policy distillation from black-box teachers.

### Stability and Objective Design

- [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/abs/2402.03898) *(2024)* — Skew-KL divergence with adaptive off-policy use of student-generated outputs; foundational OPD objective formulation (ICML 2024).
- [DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs](https://arxiv.org/abs/2503.07067) *(2025)* — Contrastive extension of skew-KL; student-generated outputs collected per epoch.
- [Veto: Stable On-Policy Distillation through Adaptive Target Reformulation](https://arxiv.org/abs/2601.07155) *(2026)* — Intermediate target distribution in logit space stabilizes training.
- [Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079) *(2026)* — Forward-KL on high-entropy teacher tokens preserves output diversity.
- [ExOPD: Learning beyond Teacher via Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125) *(2026)* — Casts OPD as dense KL-constrained RL; reward scaling enables teacher-surpassing behavior.
- [REOPOLD: Scaling Reasoning Efficiently via Relaxed On-Policy Distillation](https://arxiv.org/abs/2603.11137) *(2026)* — Relaxes imitation with reward clipping, entropy-based dynamic sampling, and explore-to-refine training.
- [PACED: Distillation at the Frontier of Student Competence](https://arxiv.org/abs/2603.11178) *(2026)* — Pass-rate weighting focuses learning on the student's competence frontier.
- [Revisiting On-Policy Distillation — Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562) *(2026)* — Truncated reverse-KL with teacher top-K support matching; fixes imbalanced signals and tokenizer mismatch.
- [Rethinking On-Policy Distillation — Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016) *(2026)* — Identifies compatible thinking patterns and novel teacher capability as OPD success conditions.
- [The Illusion of Certainty — Decoupling Capability and Calibration in OPD](https://arxiv.org/abs/2604.16830) *(2026)* — Diagnoses OPD-induced overconfidence; CaOPD replaces confidence targets with student-grounded empirical success rates.
- [Demystifying OPD — Length Inflation and Stabilization Strategies](https://arxiv.org/abs/2604.08527) *(2026)* — Repetition-driven length inflation in iterative OPD; Stable-OPD adds divergence constraints and a rollout-mixture anchor.
- [SCOPE: Signal-Calibrated On-Policy Distillation with Dual-Path Adaptive Weighting](https://arxiv.org/abs/2604.10688) *(2026)* — Routes correct student rollouts to student-PPL-weighted MLE and incorrect to teacher-PPL-weighted KL; dual-path OPD loss design.
- [HPD: Hybrid Policy Distillation for LLMs](https://arxiv.org/abs/2604.20244) *(2026)* — Unified reweighted-log-likelihood framework combining forward/reverse KL with off-policy and on-policy sampling.
- [Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe](https://arxiv.org/abs/2605.03677) *(2026)* — Offline difficulty-aware and online correctness-aware data balancing with outcome-guided margin calibration.
- [AOPD: Asymmetric On-Policy Distillation](https://arxiv.org/abs/2605.06387) *(2026)* — Replaces ineffective negative reinforcement with localized teacher-distribution matching in non-positive advantage regions.
- [vOPD: On-Policy Distillation with a Control Variate Baseline](https://arxiv.org/abs/2605.07865) *(2026)* — Closed-form per-token reverse-KL value baseline; unbiased lower-variance single-sample estimator with no extra critic.
- [Unmasking On-Policy Distillation — Where It Helps, Where It Hurts, and Why](https://arxiv.org/abs/2605.10889) *(2026)* — Training-free gradient-alignment diagnostic; best teacher flips with student capacity and task; wrong demos hurt self-distillation except on hard math.
- [The Many Faces of On-Policy Distillation — Pitfalls, Mechanisms, and Fixes](https://arxiv.org/abs/2605.11182) *(2026)* — Names three failure modes (student-prefix teacher-state mismatch, biased Top-K gradients, PI-free OPSD aggregation) and three stabilizers (stop-grad Top-K KL, RLVR teachers, SFT-stabilized students).
- [Rock Tokens — Deciphering High-Loss Tokens in On-Policy Distillation](https://arxiv.org/abs/2605.09253) *(2026)* — High-loss tokens (up to 18%) persist after apparent convergence; masking them streamlines alignment. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/YuxuanJiang1/Rock-Token)
- [BRTS: On-Policy Distillation with Best-of-N Teacher Rollout Selection](https://arxiv.org/abs/2605.09725) *(2026)* — Auxiliary teacher-context branch alongside standard OPD; selects best-of-N teacher rollouts by correctness then student-alignment. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/BWGZK-keke/BRTS)
- [Prefix Teach, Suffix Fade: Local Teachability Collapse in Strong-to-Weak On-Policy Distillation](https://arxiv.org/abs/2605.13643) *(2026)* — Dynamic release rule truncates dense supervision where the teacher's local margin collapses; counters suffix degradation in strong-to-weak OPD.
- [Beyond GRPO and On-Policy Distillation: An Empirical Sparse-to-Dense Reward Principle for LLM Post-Training](https://arxiv.org/abs/2605.12483) *(2026)* — Sparse-to-dense post-training workflow framing OPD as the dense teacher-induced reward between GRPO stages.
- [The Extrapolation Cliff in On-Policy Distillation of Near-Deterministic Structured Outputs](https://arxiv.org/abs/2605.08737) *(2026)* — Reward-extrapolation OPD collapses past a clip threshold on near-deterministic structured outputs, mapping where teacher-surpassing reward scaling stops working.
- [MOPD: Multi-Rollout On-Policy Distillation via Peer Successes and Failures](https://arxiv.org/abs/2605.12652) *(2026)* — Conditions the teacher on successful and failed peer rollouts from the student's local group, sharpening token-level supervision over independent per-rollout distillation.
- [Teacher-Guided Policy Optimization for LLM Distillation](https://arxiv.org/abs/2605.13230) *(2026)* — Feeds teacher tokens conditioned on the student's rollout as explicit on-policy-SFT targets, replacing reverse-KL's uninformative negative feedback under large teacher gaps.
- [Not All Disagreement Is Learnable: Token Teachability in On-Policy Distillation](https://arxiv.org/abs/2605.26844) *(2026)* — Applies OPD loss only to "teachable" tokens where the teacher's corrective mass lands within the student's support, separating learnable from incompatible disagreement.
- [AMR-SD: Asymmetric Meta-Reflective Self-Distillation for Token-Level Credit Assignment](https://arxiv.org/abs/2605.18529) *(2026)* — Reflection-bottlenecked privileged self-distillation converting diagnostics into ReLU-gated token-level advantages, preventing the late-stage collapse of raw-oracle conditioning.
- [Your Teacher Can't Help You Here: Combating Supervision Fidelity Decay in On-Policy Distillation](https://arxiv.org/abs/2605.30833) *(2026)* — Rewards the student's top-K candidate tokens by the teacher confidence they induce one step ahead, countering supervision-fidelity decay over long reasoning chains.
- [OPD+: Rethinking the Advantage Design for On-Policy Distillation](https://arxiv.org/abs/2606.01039) *(2026)* — Corrects on-policy distillation's biased stop-gradient advantage estimator, generalizing the objective to any f-divergence beyond the usual reverse KL.
- [SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment](https://arxiv.org/abs/2606.02530) *(2026)* — Confines reverse-KL on-policy distillation to a mined sparse subset of safety tokens, aligning behavior while sidestepping the alignment tax.

### Self-Distillation

- [OPSD: Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) *(2026)* — Single model as both teacher and student via privileged information; no external teacher.
- [SDFT: Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) *(2026)* — Demonstration-conditioned self-teaching for continual learning with less forgetting.
- [SDPO: Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) *(2026)* — Converts textual feedback into dense self-teacher signals for RL-like training.
- [Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?](https://arxiv.org/abs/2603.24472) *(2026)* — Traces failures to suppression of epistemic verbalization; task coverage determines whether conciseness helps.
- [OPSDC: On-Policy Self-Distillation for Reasoning Compression](https://arxiv.org/abs/2603.05433) *(2026)* — Compresses verbose reasoning using concise privileged self-teachers.
- [GATES: Self-Distillation under Privileged Context with Consensus Gating](https://arxiv.org/abs/2602.20574) *(2026)* — Consensus-gated asymmetric-context self-distillation without labels or rewards.
- [HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation](https://arxiv.org/abs/2603.23871) *(2026)* — Privileged self-distillation on cliff prompts where RL gradients vanish; recovers KL-regularized optimal policy.
- [RLSD: Self-Distilled RLVR](https://arxiv.org/abs/2604.03128) *(2026)* — Self-distillation as token-level credit assignment within GRPO; OPSD-style matching leaks privileged information.
- [SDZero: Self-Revision Turns Binary Rewards into Dense Supervision](https://arxiv.org/abs/2604.12002) *(2026)* — Generator-reviser dual roles; reviser converts binary feedback into token-level supervision with no external teacher.
- [OPSDL: On-Policy Self-Distillation for Long-Context Language Models](https://arxiv.org/abs/2604.17535) *(2026)* — Short-context distribution of the same model as co-evolving reverse-KL teacher under long context.
- [PBSD: Preference-Based Self-Distillation — Beyond KL Matching via Reward Regularization](https://arxiv.org/abs/2605.05040) *(2026)* — DPO-style preference learning between context-augmented teacher positives and on-policy student negatives.
- [UniSD: Towards a Unified Self-Distillation Framework for Large Language Models](https://arxiv.org/abs/2605.06597) *(2026)* — Unifies self-distillation across supervision reliability, representation alignment, and training stability. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/Ahren09/UniSD)
- [OPSD Compresses What RLVR Teaches — A Post-RL Compaction Stage](https://arxiv.org/abs/2605.06188) *(2026)* — Correct-only OPSD preserves accuracy and shortens responses; proposes SFT → RLVR → OPSD as post-RL compaction.
- [ATESD: Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning](https://arxiv.org/abs/2605.11458) *(2026)* — Treats teacher reveal ratio as a learnable control variable via Beta-policy controller with discounted learning-progress reward.
- [OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering](https://arxiv.org/abs/2605.12400) *(2026)* — Contrasts averaged teacher logits over correct vs. incorrect rollouts to form outcome-guided steering on anchor logits.
- [RLRT: Rebellious Student — Reversing Teacher Signals for Reasoning Exploration](https://arxiv.org/abs/2605.10781) *(2026)* — Upweights student tokens that diverged from teacher but still succeeded as a "valuable exploration" signal added to GRPO; +8.9% average across six math benchmarks.
- [EGRSD: Respecting Self-Uncertainty in On-Policy Self-Distillation for Efficient LLM Reasoning](https://arxiv.org/abs/2605.13255) *(2026)* — Teacher-entropy confidence gate over RLSD's direction-magnitude signal; causal-lookahead variant preserves transient pivot tokens (COLM 2026).
- [CREDIT: From Generic Correlation to Input-Specific Credit in On-Policy Self Distillation](https://arxiv.org/abs/2605.11613) *(2026)* — Recasts the self-distillation token reward as Bayesian filtering; batch-contrastive teacher baseline strips input-generic shortcuts (NeurIPS 2026).
- [OPHSD: Training with Harnesses — On-Policy Harness Self-Distillation for Complex Reasoning](https://arxiv.org/abs/2605.08741) *(2026)* — Generalizes self-distillation privileged context from a static variable (reference solution, environment trace) to a harness-driven workflow (draft-verify, plan-solve); harness is a removable training scaffold, +10.83% over OPSD on HMMT25. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation)
- [MixSD: Mixed Contextual Self-Distillation for Knowledge Injection](https://arxiv.org/abs/2605.16865) *(2026)* — Per-token Bernoulli mix of fact-conditioned and naive-conditioned base-model samples; replaces SFT for knowledge injection without collapsing held-out capability.
- [AntiSD: Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information](https://arxiv.org/abs/2605.11609) *(2026)* — Identifies the OPSD token reward as a PMI that suppresses deliberation tokens, then reverses its sign under an entropy-triggered gate. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/FloyedShen/AntiSD)
- [TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment](https://arxiv.org/abs/2605.10194) *(2026)* — Routes self-distillation KL only to annotator-marked spans to cure the all-token "distillation tax" of SDPO/SRPO.
- [AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals](https://arxiv.org/abs/2605.20643) *(2026)* — Multi-view privileged self-distillation that gates teacher-specific residuals so they can adjust update magnitude but cannot reverse the cross-view consensus direction. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/duykhuongnguyen/AVSD)
- [VPD: Learning from Language Feedback via Variational Policy Distillation](https://arxiv.org/abs/2605.15113) *(2026)* — Variational-EM self-distillation refines a feedback-conditioned self-teacher in the E-step before distilling it back via token-level KL in the M-step.
- [RMSD: Bringing Capabilities in Distribution via Relevance-Masked Self-Distillation](https://www.appliedcompute.com/research/relevance-masked-self-distillation) *(2026)* — Applied Compute's OPSD variant masking the reverse-KL loss to LLM-judge-selected behavior-relevant tokens; preserves capabilities where SFT collapses.
- [SPD: Self-Policy Distillation via Capability-Selective Subspace Projection](https://arxiv.org/abs/2605.22675) *(2026)* — Decode-time KV-subspace projection biases self-rollout generation toward capability-relevant directions, then LoRA-SFTs on those rollouts without any external verifier or teacher.
- [Multilingual Safety Alignment via Self-Distillation](https://arxiv.org/abs/2605.02971) *(2026)* — Same-model OPSD transfers English safety reasoning to low-resource languages without any response data.
- [COPSD: Crosslingual On-Policy Self-Distillation for Multilingual Reasoning](https://arxiv.org/abs/2605.09548) *(2026)* — Uses English translations and reference solutions as privileged teacher context for low-resource multilingual reasoning OPSD.
- [EDGE-OPD: Internalizing Privileged Context with Evidence Guided On-Policy Distillation](https://arxiv.org/abs/2605.23493) *(2026)* — Guides a fraction of student rollouts with the privileged context, then distills only positive-evidence tokens, internalizing rare identities OPSD never samples.
- [When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning](https://arxiv.org/abs/2605.21606) *(2026)* — Weights OPSD token supervision by within-sequence position, the strongest tested predictor of privileged-teacher reliability, rather than ambiguous teacher entropy.
- [Ditto: Reinforcing Human Behavior Simulation via Verbal Feedback](https://arxiv.org/abs/2605.20506) *(2026)* — Jointly GRPO-optimizes a draft rollout and its judge-feedback-conditioned refinement so the policy internalizes verbal guidance, targeting subjective human-simulation rather than verifiable rewards.
- [OISD: On-Policy Internal Self-Distillation of Language Models](https://arxiv.org/abs/2605.29089) *(2026)* — Distills the detached final layer into an intermediate layer across model depth via advantage-weighted Jensen–Shannon alignment — needs no privileged context.
- [ROSD: Reflective On-Policy Self-Distillation for Language Model Reasoning across Domains](https://arxiv.org/abs/2605.28014) *(2026)* — Reflection-guided OPSD restricting self-teacher distillation to a rollout's erroneous span, targeting cross-domain reasoning generalization.
- [SGSD: Skill-Conditioned Gated Self-Distillation for LLM Reasoning](https://arxiv.org/abs/2605.28791) *(2026)* — Skill-conditioned OPSD whose retrieved-skill teachers are outcome-validated before distillation, extending privileged self-distillation to unreliable experience-derived context.
- [Distilling LLM Feedback for Lean Theorem Proving](https://arxiv.org/abs/2605.30861) *(2026)* — Distills a self-teacher conditioned on LLM-generated critique of the student's attempt, injecting external knowledge through natural-language feedback rather than logits or solutions.
- [CAST: Non-Privileged Clipped Asymmetric Self-Teaching with Advantage Flipping for GRPO](https://arxiv.org/abs/2606.00172) *(2026)* — Answer-free correctness-conditioned self-teacher bidirectionally flips GRPO token-advantage signs, unlike the privileged-context teachers of related self-distillation methods.
- [SC-SDPO: Restoring the Sweet Spot via Pass-Rate Weighted Self-Distillation](https://arxiv.org/abs/2605.27765) *(2026)* — Reweights SDPO's self-distillation loss by an on-the-fly pass-rate term, restoring the difficulty sweet spot that pure self-distillation discards.
- [Self-Supervised On-Policy Distillation for Reasoning Language Models](https://arxiv.org/abs/2605.17497) *(2026)* — Conditions a self-teacher on a successful peer completion to densely supervise failed on-policy prefixes within each GRPO group.
- [Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation](https://arxiv.org/abs/2605.15239) *(2026)* — On-policy self-distillation for safety using a privileged-context self-teacher, with flip-rate prompt search selecting contexts that activate latent refusal.
- [Internalize the Temperature: On-Policy Self-Distillation as Policy Reheater for Reinforcement Learning](https://arxiv.org/abs/2606.00755) *(2026)* — Distills a temperature-scaled copy of the model's own logits to restore entropy in RL-collapsed policies before continued training.

### Context and Experience Internalization

- [OPCD: On-Policy Context Distillation for Language Models](https://arxiv.org/abs/2602.12275) *(2026)* — Context-conditioned teacher on student rollouts; distills system prompts and experiential knowledge.
- [OEL: Online Experiential Learning for Language Models](https://arxiv.org/abs/2603.16856) *(2026)* — Deployment loop using OPCD for consolidating interaction traces into weights.
- [Aligning Language Models from User Interactions](https://arxiv.org/abs/2603.12273) *(2026)* — Hindsight self-distillation from user follow-ups; same model conditioned on the follow-up serves as the teacher.
- [MAIGO: Mitigating Lost-in-Conversation with History-Cleaned On-Policy Self-Distillation](https://arxiv.org/abs/2605.27186) *(2026)* — History-cleaned OPSD distilling assistant-stripped reference distributions onto the student's own sharded rollouts, fixing lost-in-conversation self-contamination.
- [Found in Conversation: LLMs Teach Themselves to Close the Multi-Turn Gap](https://arxiv.org/abs/2605.24432) *(2026)* — View-asymmetric self-distillation aligning on-policy multi-turn trajectories to the same model's single-turn behavior, needing no external teacher.
- [Same Evidence, Different Answers: Canonical-Context On-Policy Distillation for Multi-Turn Language Models](https://arxiv.org/abs/2605.30251) *(2026)* — Canonical-context OPSD aligning multi-turn student trajectories to a full-context frozen self-teacher, countering self-anchored drift across turns.
- [Weak Critics Make Strong Learners: On-Policy Critique Distillation for Scalable Oversight](https://arxiv.org/abs/2606.00424) *(2026)* — Distills a weak-critic-conditioned self-teacher into a critique-free student, letting a weaker overseer improve a stronger model without test-time critiques.
- [Reasoning Compression with Mixed-Policy Distillation](https://arxiv.org/abs/2605.08776) *(2026)* — A larger teacher rewrites student-sampled reasoning into concise traces for KL alignment, transferring brevity instead of enforcing length penalties.

### Efficiency, Systems, and Privacy Variants

- [Prefix OPD: Fast and Effective On-policy Distillation from Reasoning Prefixes](https://arxiv.org/abs/2602.15260) *(2026)* — Distills only reasoning prefixes, cutting training FLOPs 2×-47×.
- [OVD: On-policy Verbal Distillation](https://arxiv.org/abs/2601.21968) *(2026)* — Trajectory-level verbal scoring instead of token-level logit matching; relaxes alignment requirements.
- [pi-Distill: Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942) *(2026)* — Training-time privileged information in agentic settings where only actions are observable.
- [Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline OPD](https://arxiv.org/abs/2604.13010) *(2026)* — Precomputes teacher log-probs once over SFT rollouts; 4× speedup via teacher-consistency condition.
- [DP-OPD: Differentially Private On-Policy Distillation for Language Models](https://arxiv.org/abs/2604.04461) *(2026)* — Student-rollout OPD with DP-SGD on student updates; first OPD recipe with sample-level differential privacy.
- [TIP: Token Importance in On-Policy Distillation](https://arxiv.org/abs/2604.14084) *(2026)* — Selective training on high-entropy and confidently-wrong low-entropy tokens; matches full-token baselines at lower memory.
- [Nitrobrew: Communication- and Memory-Efficient On-Policy Distillation](https://blog.tilderesearch.com/blog/nitrobrew) *(2026)* — Hidden-state teacher→student transport plus tile-wise online divergence kernel; 1.5-3× throughput.
- [NPD: Near-Policy Distillation via Asynchronous Generation and Selective Packing](https://arxiv.org/abs/2605.05940) *(2026)* — Decouples generation from training; sparse updates plus Δ-IFD filtering; 8.1× speedup over on-policy baselines.
- [Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning](https://arxiv.org/abs/2605.07804) *(2026)* — Top-k overlap monitors prefix drift; attenuates unreliable rewards and truncates drifted rollouts.
- [EffOPD: Learning to Foresee — Unlocking Efficiency of On-Policy Distillation](https://arxiv.org/abs/2605.11739) *(2026)* — Adaptively extrapolates along the current update step for ~3× training acceleration with no extra trainables.
- [Less is More: Early Stopping Rollout for On-Policy Distillation](https://arxiv.org/abs/2605.27028) *(2026)* — Restricts rollout and reverse-KL loss to the first response tokens, where teacher supervision is strongest before it decays toward the student baseline.
- [ADWIN: Adaptive Windows for Horizon-Aware On-Policy Distillation](https://arxiv.org/abs/2605.28396) *(2026)* — Trains OPD on short teacher-anchored prefix windows whose horizon is adapted online by delayed full-rollout probes auditing prefix–full gradient alignment.
- [Are Full Rollouts Necessary for On-Policy Distillation?](https://arxiv.org/abs/2605.31490) *(2026)* — Controls OPD rollout horizon by progressively expanding or permanently truncating student rollouts, distilling only reliable early segments to cut compute.
- [f-OPD: Stabilizing Long-Horizon On-Policy Distillation with Freshness-Aware Control](https://arxiv.org/abs/2605.17862) *(2026)* — Scores per-sample freshness from rollout–supervision drift to stabilize asynchronous on-policy distillation where generation and training are decoupled.

## Taxonomy

Cross-cutting views over the canonical papers. Many entries span multiple categories — this is for orientation, not strict partitioning.

### By Teacher Type

| Teacher Type | Papers |
|---|---|
| External white-box | [MiniLLM](https://arxiv.org/abs/2306.08543), [GKD](https://arxiv.org/abs/2306.13649), [DistiLLM](https://arxiv.org/abs/2402.03898), [DistiLLM-2](https://arxiv.org/abs/2503.07067), [Veto](https://arxiv.org/abs/2601.07155), [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079), [ExOPD](https://arxiv.org/abs/2602.12125), [REOPOLD](https://arxiv.org/abs/2603.11137), [PACED](https://arxiv.org/abs/2603.11178), [Prefix OPD](https://arxiv.org/abs/2602.15260), [Revisiting OPD](https://arxiv.org/abs/2603.25562), [Rethinking OPD](https://arxiv.org/abs/2604.13016), [Lightning OPD](https://arxiv.org/abs/2604.13010), [Uni-OPD](https://arxiv.org/abs/2605.03677), [SOD](https://arxiv.org/abs/2605.07725), [AOPD](https://arxiv.org/abs/2605.06387), [vOPD](https://arxiv.org/abs/2605.07865), [SCOPE](https://arxiv.org/abs/2604.10688), [HPD](https://arxiv.org/abs/2604.20244), [TIP](https://arxiv.org/abs/2604.14084), [DP-OPD](https://arxiv.org/abs/2604.04461), [NPD](https://arxiv.org/abs/2605.05940), [Prune-OPD](https://arxiv.org/abs/2605.07804), [EffOPD](https://arxiv.org/abs/2605.11739), [CoDistill-GRPO](https://arxiv.org/abs/2605.08873), [Rock Tokens](https://arxiv.org/abs/2605.09253), [Sparse-to-Dense](https://arxiv.org/abs/2605.12483), [MOTAB](https://arxiv.org/abs/2605.19433), [TGPO](https://arxiv.org/abs/2605.13230), [TA-OPD](https://arxiv.org/abs/2605.26844), [ESR](https://arxiv.org/abs/2605.27028), [ADWIN](https://arxiv.org/abs/2605.28396), [dGRPO](https://arxiv.org/abs/2605.12227), [LGR](https://arxiv.org/abs/2605.30833), [TRB](https://arxiv.org/abs/2605.31159), [POPD/TOPD](https://arxiv.org/abs/2605.31490), [f-OPD](https://arxiv.org/abs/2605.17862), [NF-OPD](https://arxiv.org/abs/2606.00305), [OPD+](https://arxiv.org/abs/2606.01039), [TrOPD](https://arxiv.org/abs/2606.01249), [MPD](https://arxiv.org/abs/2605.08776) |
| External black-box | [Black-Box OPD / GAD](https://arxiv.org/abs/2511.10643), [OVD](https://arxiv.org/abs/2601.21968), [ROPD](https://arxiv.org/abs/2605.07396), [OmniOPD](https://arxiv.org/abs/2606.01476) |
| Self-teacher with privileged context | [OPSD](https://arxiv.org/abs/2601.18734), [SDFT](https://arxiv.org/abs/2601.19897), [SDPO](https://arxiv.org/abs/2601.20802), [OPSDC](https://arxiv.org/abs/2603.05433), [GATES](https://arxiv.org/abs/2602.20574), [pi-Distill](https://arxiv.org/abs/2602.04942), [RLSD](https://arxiv.org/abs/2604.03128), [SDZero](https://arxiv.org/abs/2604.12002), [OGLS-SD](https://arxiv.org/abs/2605.12400), [PBSD](https://arxiv.org/abs/2605.05040), [UniSD](https://arxiv.org/abs/2605.06597), [ATESD](https://arxiv.org/abs/2605.11458), [RLRT](https://arxiv.org/abs/2605.10781), [EGRSD](https://arxiv.org/abs/2605.13255), [CREDIT](https://arxiv.org/abs/2605.11613), [SDAR](https://arxiv.org/abs/2605.15155), [MixSD](https://arxiv.org/abs/2605.16865), [AntiSD](https://arxiv.org/abs/2605.11609), [TRACE](https://arxiv.org/abs/2605.10194), [AVSD](https://arxiv.org/abs/2605.20643), [VPD](https://arxiv.org/abs/2605.15113), [RMSD](https://www.appliedcompute.com/research/relevance-masked-self-distillation), [SPD](https://arxiv.org/abs/2605.22675), [MSD-Safety](https://arxiv.org/abs/2605.02971), [COPSD](https://arxiv.org/abs/2605.09548), [EDGE-OPD](https://arxiv.org/abs/2605.23493), [EMPO²](https://arxiv.org/abs/2602.23008), [StepOPSD](https://arxiv.org/abs/2605.27140), [PW-OPSD](https://arxiv.org/abs/2605.21606), [Ditto](https://arxiv.org/abs/2605.20506), [ROSD](https://arxiv.org/abs/2605.28014), [SGSD](https://arxiv.org/abs/2605.28791), [AMR-SD](https://arxiv.org/abs/2605.18529), [Feedback Distillation](https://arxiv.org/abs/2605.30861), [SC-SDPO](https://arxiv.org/abs/2605.27765), [SSOPD](https://arxiv.org/abs/2605.17497), [OPSA](https://arxiv.org/abs/2605.15239), [OPCritD](https://arxiv.org/abs/2606.00424) |
| Internal self-teacher (cross-depth) | [OISD](https://arxiv.org/abs/2605.29089) |
| Self-teacher (non-privileged / answer-free) | [CAST](https://arxiv.org/abs/2606.00172), [TS-OPSD](https://arxiv.org/abs/2606.00755), [SafeSteer](https://arxiv.org/abs/2606.02530) |
| Context-conditioned | [OPCD](https://arxiv.org/abs/2602.12275), [OEL](https://arxiv.org/abs/2603.16856), [Multi-Rollout MOPD](https://arxiv.org/abs/2605.12652), [MAIGO](https://arxiv.org/abs/2605.27186), [FiC](https://arxiv.org/abs/2605.24432), [CCOPD](https://arxiv.org/abs/2605.30251) |
| Multiple / lifecycle teachers | [MiMo-V2-Flash MOPD](https://arxiv.org/abs/2601.02780), [GLM-5](https://arxiv.org/abs/2602.15763), [Qwen3](https://arxiv.org/abs/2505.09388), [Baichuan-M3](https://arxiv.org/abs/2602.06570), [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf), [CoPD](https://arxiv.org/abs/2604.27083), [MAD-OPD](https://arxiv.org/abs/2605.01347), [KAT-Coder-V2](https://arxiv.org/abs/2603.27703), [CaMOPD](https://arxiv.org/abs/2605.27115), [CollectionLoRA](https://arxiv.org/abs/2605.25378) |

### By Primary Goal

| Goal | Papers |
|---|---|
| Compression / strong-to-weak transfer | [MiniLLM](https://arxiv.org/abs/2306.08543), [GKD](https://arxiv.org/abs/2306.13649), [Qwen3](https://arxiv.org/abs/2505.09388), [Prefix OPD](https://arxiv.org/abs/2602.15260), [Rethinking OPD](https://arxiv.org/abs/2604.13016), [Lightning OPD](https://arxiv.org/abs/2604.13010), [MOTAB](https://arxiv.org/abs/2605.19433), [ADWIN](https://arxiv.org/abs/2605.28396), [TRB](https://arxiv.org/abs/2605.31159), [POPD/TOPD](https://arxiv.org/abs/2605.31490), [NF-OPD](https://arxiv.org/abs/2606.00305), [OPD+](https://arxiv.org/abs/2606.01039), [TrOPD](https://arxiv.org/abs/2606.01249) |
| Post-RL consolidation / skill integration | [MiMo MOPD](https://arxiv.org/abs/2601.02780), [GLM-5](https://arxiv.org/abs/2602.15763), [ExOPD](https://arxiv.org/abs/2602.12125), [CoPD](https://arxiv.org/abs/2604.27083), [OPCritD](https://arxiv.org/abs/2606.00424), [TS-OPSD](https://arxiv.org/abs/2606.00755) |
| Continual learning | [SDFT](https://arxiv.org/abs/2601.19897), [OPCD](https://arxiv.org/abs/2602.12275), [OEL](https://arxiv.org/abs/2603.16856), [MixSD](https://arxiv.org/abs/2605.16865), [EDGE-OPD](https://arxiv.org/abs/2605.23493), [CaMOPD](https://arxiv.org/abs/2605.27115), [MAIGO](https://arxiv.org/abs/2605.27186), [FiC](https://arxiv.org/abs/2605.24432), [CCOPD](https://arxiv.org/abs/2605.30251) |
| RL replacement / augmentation | [SDPO](https://arxiv.org/abs/2601.20802), [RLTF-SD](https://arxiv.org/abs/2602.02482), [RLAD](https://arxiv.org/abs/2602.22495), [REOPOLD](https://arxiv.org/abs/2603.11137), [RLSD](https://arxiv.org/abs/2604.03128), [SDZero](https://arxiv.org/abs/2604.12002), [OGLS-SD](https://arxiv.org/abs/2605.12400), [PBSD](https://arxiv.org/abs/2605.05040), [CoDistill-GRPO](https://arxiv.org/abs/2605.08873), [RLRT](https://arxiv.org/abs/2605.10781), [EGRSD](https://arxiv.org/abs/2605.13255), [CREDIT](https://arxiv.org/abs/2605.11613), [SDAR](https://arxiv.org/abs/2605.15155), [Sparse-to-Dense](https://arxiv.org/abs/2605.12483), [AntiSD](https://arxiv.org/abs/2605.11609), [TRACE](https://arxiv.org/abs/2605.10194), [AVSD](https://arxiv.org/abs/2605.20643), [VPD](https://arxiv.org/abs/2605.15113), [RMSD](https://www.appliedcompute.com/research/relevance-masked-self-distillation), [Multi-Rollout MOPD](https://arxiv.org/abs/2605.12652), [EMPO²](https://arxiv.org/abs/2602.23008), [StepOPSD](https://arxiv.org/abs/2605.27140), [TGPO](https://arxiv.org/abs/2605.13230), [Ditto](https://arxiv.org/abs/2605.20506), [OISD](https://arxiv.org/abs/2605.29089), [ROSD](https://arxiv.org/abs/2605.28014), [SGSD](https://arxiv.org/abs/2605.28791), [AMR-SD](https://arxiv.org/abs/2605.18529), [dGRPO](https://arxiv.org/abs/2605.12227), [Feedback Distillation](https://arxiv.org/abs/2605.30861), [CAST](https://arxiv.org/abs/2606.00172), [SC-SDPO](https://arxiv.org/abs/2605.27765), [f-OPD](https://arxiv.org/abs/2605.17862), [SSOPD](https://arxiv.org/abs/2605.17497), [OPSA](https://arxiv.org/abs/2605.15239), [SafeSteer](https://arxiv.org/abs/2606.02530) |
| Reasoning compression | [OPSDC](https://arxiv.org/abs/2603.05433), [MPD](https://arxiv.org/abs/2605.08776) |
| Black-box distillation | [GAD](https://arxiv.org/abs/2511.10643), [OVD](https://arxiv.org/abs/2601.21968), [ROPD](https://arxiv.org/abs/2605.07396), [OmniOPD](https://arxiv.org/abs/2606.01476) |

## Adjacent and Enabling Work

Papers that are not canonical OPD but matter for understanding or deploying it.

### Cross-Tokenizer and Model-Family Enablers

- [ULD: Towards Cross-Tokenizer Distillation](https://arxiv.org/abs/2402.12030) *(2024)* — Universal Logit Distillation; foundational enabler for cross-family OPD.
- [Multi-Level OT for Universal Cross-Tokenizer KD](https://arxiv.org/abs/2412.14528) *(2024)* — Token- and sequence-level optimal transport for cross-tokenizer KD.
- [CDM: Enhancing Cross-Tokenizer KD with Contextual Dynamical Mapping](https://arxiv.org/abs/2502.11104) *(2025)* — Contextual dynamic mapping for vocabulary alignment.
- [Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083) *(2025)* — Approximate likelihood matching across fundamentally different tokenizers.
- [Cross-Tokenizer Likelihood Scoring Algorithms](https://arxiv.org/abs/2512.14954) *(2025)* — Exact and approximate sequence likelihood scoring across BPE vocabularies.
- [DSKD: A Dual-Space Framework for General KD](https://arxiv.org/abs/2504.11426) *(2025)* — Unifies output spaces; supports on- and off-policy KD between any two LLMs.
- [GOLD: Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) *(2025)* — Cross-tokenizer OPD with TRL integration.
- [CTPD: Cross Tokenizer Preference Distillation](https://arxiv.org/abs/2601.11865) *(2026)* — Aligned-span projection plus teacher-anchored DPO with cross-tokenizer importance sampling.
- [DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer KD](https://arxiv.org/abs/2602.21669) *(2026)* — Dual-space token weighting plus Soft-DTW differentiable sequence alignment.
- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466) *(2026)* — Byte-level conversion of teacher distributions plus byte-level student decoder for mismatched tokenizers.
- [SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation](https://arxiv.org/abs/2605.07711) *(2026)* — Short multi-token continuations replace exact matching; recovers teacher signal at mismatched positions.

### Mismatch Mitigation and Student Quality

- [Exploring and Enhancing Distribution Transfer in KD](https://arxiv.org/abs/2409.12512) *(2024)* — Analyzes reverse-KL with student-generated output; proposes OKD.
- [FIRST: Efficient Trustworthy Distillation](https://arxiv.org/abs/2408.12168) *(2024)* — Teacher recalibration for trustworthy offline KD.
- [Multi-Granularity Semantic Revision](https://arxiv.org/abs/2407.10068) *(2024)* — Sequence correction for low-quality student-generated outputs.
- [Warmup-Distill](https://arxiv.org/abs/2502.11766) *(2025)* — Bridges distribution mismatch before distillation begins.
- [TAID: Temporally Adaptive Interpolated Distillation](https://arxiv.org/abs/2501.16937) *(2025)* — Addresses teacher-student mismatch via adaptive interpolation.
- [SpecKD: Speculative Decoding for Effective KD](https://arxiv.org/abs/2510.24021) *(2025)* — Speculative-decoding-inspired selective token-level losses.
- [Knowledge Distillation with Training Wheels](https://arxiv.org/abs/2502.17717) *(2025)* — Entropy-regularized value optimization with on-/off-policy demonstrations.
- [Revealing the Power of Post-Training via KD](https://arxiv.org/abs/2509.26497) *(2025)* — Offline on-policy KD: student generates, then teacher labels.
- [TSD-KD: Explain in Your Own Words](https://arxiv.org/abs/2603.13260) *(2026)* — Student proposes candidates, teacher reranks, selective token distillation.
- [SSD: Embarrassingly Simple Self-Distillation Improves Code Generation](https://arxiv.org/abs/2604.01193) *(2026)* — Temperature-shifted self-sampling plus SFT; identifies precision-exploration conflict.
- [AdaSwitch: Balancing Exploration and Guidance in KD via Adaptive Switching](https://arxiv.org/abs/2510.07842) *(2025)* — Switches between on-policy rollouts and off-policy teacher data via context-aware divergence threshold.
- [DDT: Towards On-Policy SFT via Distribution Discriminant Theory](https://arxiv.org/abs/2602.12222) *(2026)* — In-Distribution Finetuning and Hinted Decoding realign training data to the student's distribution.
- [DASD: Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning](https://arxiv.org/abs/2601.09088) *(2026)* — On-policy correction pipeline for distribution mismatch and exposure bias in sequence-level CoT distillation.
- [Distillation Traps and Guards: A Calibration Knob for LLM Distillability](https://arxiv.org/abs/2604.18963) *(2026)* — Post-hoc calibrates teachers via RFT to control distillability against tail noise and instability.
- [A Predictive Law for On-Policy Self-Distillation From World Feedback](https://arxiv.org/abs/2605.30070) *(2026)* — Predictive law: a linear relation between the initial student–self-teacher gap and final OPSD improvement, estimable before training.

### Preference, Reward-Guided, and Hybrid RL+KD

- [Direct Preference Knowledge Distillation](https://arxiv.org/abs/2406.19774) *(2024)* — Preference-aware KD combining reverse-KL with implicit reward objectives.
- [Online Knowledge Distillation with Reward Guidance](https://arxiv.org/abs/2505.18952) *(2025)* — Sequential KD via preference optimization; offline and online variants.
- [KDRL](https://arxiv.org/abs/2506.02208) *(2025)* — Unified reverse-KL KD with RL in a single post-training objective.
- [RLTF-SD: Expanding RL via Text Feedback](https://arxiv.org/abs/2602.02482) *(2026)* — Internalizes text feedback via self-distillation.
- [RLAD: Reinforcement-aware KD for LLM Reasoning](https://arxiv.org/abs/2602.22495) *(2026)* — Trust-region ratio distillation on student rollouts.
- [Multi-Token Prediction via Self-Distillation](https://arxiv.org/abs/2602.06019) *(2026)* — Online self-distillation for multi-token prediction and faster inference.
- [ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](https://arxiv.org/abs/2509.25100) *(2025)* — Mixed-policy preference distillation with student-generated outputs; black-box cross-architecture transfer.
- [SRPO: Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing](https://arxiv.org/abs/2604.02288) *(2026)* — Routes correct student rollouts to reward-based RL and failed ones to self-distillation.
- [KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation](https://arxiv.org/abs/2504.19024) *(2025)* — K-step Bellman return replaces high-variance single-step REINFORCE in sequence-level OPD.
- [Rethinking LLM Distillation: A Constrained MDP Perspective](https://arxiv.org/abs/2509.22921) *(2025)* — Maximizes task reward under hard KL constraint against the teacher; avoids manual Lagrangian tuning.
- [RLKD: Distilling LLMs' Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.16142) *(2025)* — Generative Structure Reward Model on student rollouts; outperforms SFT-RL pipelines on 0.1% data.
- [LUFFY: Learning to Reason under Off-Policy Guidance](https://arxiv.org/abs/2504.14945) *(2025)* — Mixed-policy GRPO combining on-policy rollouts with off-policy teacher traces via regularized importance sampling.
- [BOND: Aligning LLMs with Best-of-N Distillation](https://arxiv.org/abs/2407.14622) *(2024)* — RL mimicking best-of-N via Jeffreys-divergence matching; eliminates inference-time BoN cost.
- [Faster WIND: Accelerating Iterative Best-of-N Distillation for LLM Alignment](https://arxiv.org/abs/2410.20727) *(2024)* — Game-theoretic iterative BoN as self-play; win-rate dominance optimization (AISTATS 2025).
- [AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](https://arxiv.org/abs/2503.02832) *(2025)* — Casts RLHF as token-level distillation by injecting DPO rewards (ACL 2025).
- [KEPO: Knowledge-Enhanced Preference Optimization for Reinforcement Learning with Reasoning](https://arxiv.org/abs/2602.00400) *(2026)* — Quality-gated OPD on high-quality trajectories plus knowledge-enhanced exploration via teacher hints.
- [𝒳-KD: General Experiential Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2602.12674) *(2026)* — Jointly models teacher reward and policy-distills so the student learns inside the teacher's original environment.
- [ExGRPO: Probing to Refine — Reinforcement Distillation of LLMs via Explanatory Inversion](https://arxiv.org/abs/2603.19266) *(2026)* — Explanatory probes plus dialogue-structure utility bonus reward coherent reasoning over memorized answers.
- [NPO: Near-Future Policy Optimization](https://arxiv.org/abs/2604.20733) *(2026)* — Later checkpoint of same policy as teacher; AutoNPO adaptively triggers switch to maximize RLVR signal.
- [CoDistill-GRPO: A Co-Distillation Recipe for Efficient Group Relative Policy Optimization](https://arxiv.org/abs/2605.08873) *(2026)* — Dual GRPO with on-policy KD reward between large/small models; matches standard GRPO with 18% speedup.
- [Combining On-Policy Optimization and Distillation for Long-Context Reasoning in Large Language Models](https://arxiv.org/abs/2605.12227) *(2026)* — dGRPO augments GRPO with dense teacher-KL guidance on student rollouts in one objective, for long-context reasoning.

### Self-Play and Iterative Bootstrapping

- [SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) *(2024)* — Self-play distinguishing own generations from human references (ICML 2024).
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) *(2024)* — Iterative DPO with model-as-judge self-rewards on own generations.
- [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://arxiv.org/abs/2501.04519) *(2025)* — MCTS-guided self-evolution; policy and PRM co-improve via code-augmented reasoning.
- [rStar2-Agent: Agentic Reasoning Technical Report](https://arxiv.org/abs/2508.20722) *(2025)* — GRPO with Resample-on-Correct rollouts plus multi-stage SFT→RL recipe for 14B agentic reasoner.
- [π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data](https://arxiv.org/abs/2604.14054) *(2026)* — Examiner-generated tasks plus question-construction-paths as privileged context for dense student supervision.
- [SPHERE: Self-Evolved Preference Optimization for Mathematical Reasoning in SLMs](https://arxiv.org/abs/2503.04813) *(2025)* — PRM/ORM-scored MCTS rollouts plus self-correction yield preference pairs for iterative DPO.
- [SGS: Scaling Self-Play with Self-Guidance](https://arxiv.org/abs/2604.20209) *(2026)* — Three-role self-play (Solver, Generator, Reviewer) for theorem proving; 7B beats 671B at pass@4 on Lean4.
- [Self-Verified Distillation: Your Language Model Is Secretly Its Own Synthetic Data Pipeline](https://arxiv.org/abs/2605.26132) *(2026)* — Samples candidate solutions to unlabeled questions, filters them through a multi-stage self-verification cascade, then SFTs on accepted ones — no teacher or ground-truth.
- [IRIS: Interpolative Rényi Iterative Self-play for Large Language Model Fine-Tuning](https://arxiv.org/abs/2604.20933) *(2026)* — Generalizes SPIN-style self-play with an adaptively scheduled Rényi-family objective over annotated versus self-generated responses, unifying several self-play variants.
- [Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](https://arxiv.org/abs/2601.08198) *(2026)* — Replaces SPIN's pairwise objective with a triplet over annotated, current-synthetic, and initial-policy responses to stabilize iterative self-play.

### Precursors

- [Autoregressive KD through Imitation Learning](https://arxiv.org/abs/2009.07253) *(2020)* — Early precursor framing sequence-model KD as imitation learning.
- [Learning by Distilling Context](https://arxiv.org/abs/2209.15189) *(2022)* — Context distillation; key precursor to OPCD and OEL.

## Domain Extensions

OPD applied to non-text-reasoning settings — agents, multimodal models, diffusion, audio, robotics — and to inference acceleration via speculative decoding. These pass the inclusion criterion (student rollouts central to the learning signal) but on substrates beyond LLM text reasoning.

### Agent, Multimodal, and Other Extensions

- [Structured Agent Distillation](https://arxiv.org/abs/2505.13820) *(2025)* — Queries teacher online to avoid distribution drift in agent settings.
- [From Deferral to Learning: Online In-Context KD for LLM Cascades](https://arxiv.org/abs/2509.22984) *(2025)* — Teacher-student cascade with reusable online knowledge store.
- [AllMem](https://arxiv.org/abs/2602.13680) *(2026)* — Offline on-policy distillation for long-context modeling.
- [Video-OPD](https://arxiv.org/abs/2602.02994) *(2026)* — OPD for temporal video grounding in multimodal LLMs.
- [Reinforced Attention Learning](https://arxiv.org/abs/2602.04884) *(2026)* — On-policy attention distillation for multimodal models.
- [SCoRe: From Correction to Mastery via Reinforced Distillation of LLM Agents](https://arxiv.org/abs/2509.14257) *(2025)* — Teacher intervenes at first critical error in student agent trajectories for corrective distillation.
- [VOLD: Reasoning Transfer from LLMs to Vision-Language Models via OPD](https://arxiv.org/abs/2510.23497) *(2025)* — Text-only teacher distills reasoning into VLM via student-generated traces with combined GRPO and OPD.
- [X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs](https://arxiv.org/abs/2603.24596) *(2026)* — Student on-policy rollouts with token-level teacher feedback for cross-modal speech-LLM distillation.
- [VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via OPD](https://arxiv.org/abs/2603.26666) *(2026)* — Reverse-KL OPD bridging offline SFT and online RL for robotic manipulation.
- [TCOD: Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents](https://arxiv.org/abs/2604.24005) *(2026)* — Short-to-long trajectory-depth curriculum mitigating multi-turn KL instability.
- [LLM4Teach: Large Language Model as a Policy Teacher for Training RL Agents](https://arxiv.org/abs/2311.13373) *(2023)* — LLM teacher distills into small RL agent that surpasses teacher through environment interaction.
- [RPD: Refined Policy Distillation — From VLA Generalists to RL Experts](https://arxiv.org/abs/2503.05833) *(2025)* — Teacher VLA actions guide student during RL exploration; combines RL with behavioral cloning (IROS 2026).
- [π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](https://arxiv.org/abs/2510.14974) *(2025)* — Imitation distillation aligns student flow-model trajectories with teacher under standard flow matching (ICLR 2026).
- [Step-Audio-R1 Technical Report](https://arxiv.org/abs/2511.15848) *(2025)* — Modality-Grounded Reasoning Distillation produces audio reasoning grounded in acoustic features.
- [OPD-AVMP: On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning](https://arxiv.org/abs/2604.07944) *(2026)* — Generalized OPD for LLM-based driving planners; 5× compression at near-teacher performance.
- [CORD: Bridging the Audio–Text Reasoning Gap via Weighted On-policy Cross-modal Distillation](https://arxiv.org/abs/2601.16547) *(2026)* — Audio-conditioned rollouts; text-conditioned same model as teacher; importance-weighted reverse KL plus GRPO.
- [Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents](https://arxiv.org/abs/2604.10674) *(2026)* — Plain-prompt student; skill-augmented same model as token-level self-teacher for multi-turn agent training.
- [CoPD: Co-Evolving Policy Distillation](https://arxiv.org/abs/2604.27083) *(2026)* — Parallel expert training with bidirectional OPD; experts co-evolve as mutual teachers during RLVR.
- [PRISM: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL](https://arxiv.org/abs/2604.28123) *(2026)* — Black-box OPD pre-alignment between SFT and RLVR for VLMs; MoE discriminator supplies adversarial signals.
- [D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models](https://arxiv.org/abs/2605.05204) *(2026)* — OPSD ported to few-step T2I diffusion; text-only student vs. text+image teacher with velocity-MSE on rollouts.
- [Flow-OPD: On-Policy Distillation for Flow Matching Models](https://arxiv.org/abs/2605.08063) *(2026)* — Per-domain Flow-GRPO experts supervise student SDE rollouts via reverse-KL with Manifold Anchor Regularization.
- [VISD: Enhancing Video Reasoning via Structured Self-Distillation](https://arxiv.org/abs/2605.06094) *(2026)* — Structured video judge feeds EMA teacher with privileged feedback; direction-magnitude decoupling stabilizes RL+supervision.
- [TAD: Temporal-Aware Trajectory Self-Distillation for Fast and Accurate Diffusion LLM](https://arxiv.org/abs/2605.09536) *(2026)* — Partitions masked positions by remaining decoding steps into near (CE) and distant (KL) subsets. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/BHmingyang/TAD)
- [DiMO: Distilling Masked Diffusion Models into One-step Generator](https://arxiv.org/abs/2503.15457) *(2025)* — First OPD for masked discrete diffusion image generation; Generalized Jeffrey divergence with DMD-style auxiliary (ICCV 2025).
- [SDAR: Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155) *(2026)* — Sigmoid-gated OPSD auxiliary on top of GRPO for multi-turn agents; amplifies positive-gap, attenuates negative-gap tokens (COLM 2026). [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/ZJU-REAL/SDAR)
- [GEAR: Granularity-Adaptive Advantage Reweighting for LLM Agents via Self-Distillation](https://arxiv.org/abs/2605.11853) *(2026)* — Reverse-KL between student and ground-truth-conditioned same-model teacher gives token saliency; KL-initiated entropy-terminated segmentation propagates credit within segments to sign-aware reweight GRPO advantages.
- [Revisiting DAgger in the Era of LLM-Agents](https://arxiv.org/abs/2605.12913) *(2026)* — Turn-level (DAgger) and trajectory-prefix (AggreVaTe) student/teacher rollout mixtures with teacher actions queried at every visited state.
- [DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models](https://arxiv.org/abs/2605.15055) *(2026)* — Lifts OPD from autoregressive tokens to diffusion denoising via closed-form reverse-KL along student rollouts; unifies SDE and ODE samplers.
- [Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning](https://arxiv.org/abs/2605.22511) *(2026)* — Alternates GRPO with offline self-distillation on correct-fewest-search / divergent-sibling pairs mined from the converged rollout pool.
- [Healthcare AI GYM for Medical Agents](https://arxiv.org/abs/2605.02943) *(2026)* — Clinical-agent gymnasium plus Turn-level Truncated OPD: an EMA teacher conditioned on outcome-privileged hints that are stripped before logprob comparison.
- [HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents](https://arxiv.org/abs/2605.07177) *(2026)* — Parallel multimodal search agent that applies OPD only to failed rollouts to salvage correct intermediate tool calls from GRPO's uniform negative advantage. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/DeepExperience/HyperEyes)
- [DGPO: Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](https://arxiv.org/abs/2508.20324) *(2025)* — Selective reverse-KL teacher guidance on student-generated outputs within PPO; teacher intervenes only when the compact agent's autonomous attempts fail.
- [LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation](https://arxiv.org/abs/2512.23576) *(2025)* — On-policy distillation for real-time interactive video diffusion, extending the Self-Forcing few-step student-rollout recipe.
- [EMPO²: Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](https://arxiv.org/abs/2602.23008) *(2026)* — Hybrid agent RL whose off-policy mode distills memory-tip-conditioned rollouts into the tips-free policy, internalizing memory-driven exploration without tips at inference.
- [AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://arxiv.org/abs/2605.13724) *(2026)* — On-policy flow-map distillation along the student's own Euler rollout; supports any-step video generation.
- [StepOPSD: Step-Aware Online Preference Distillation for Agent Reinforcement Learning](https://arxiv.org/abs/2605.27140) *(2026)* — Converts teacher–student log-probability gaps into sign-preserving GRPO advantage shaping localized to action-centered step spans rather than whole agent trajectories.
- [What and When to Distill: Selective Hindsight Distillation for Multi-Turn Agents](https://arxiv.org/abs/2605.19447) *(2026)* — An environment-conditioned teacher's hindsight log-probability gap on student action tokens becomes a bounded, sign-aware GRPO advantage reweight anchored to task reward.
- [Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation](https://arxiv.org/abs/2605.18740) *(2026)* — Distills a crop-conditioned privileged self-teacher into the full-image student along the student's own multimodal rollouts, internalizing regional-to-global visual zooming.
- [Visual-Advantage On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2605.21924) *(2026)* — Reweights VLM on-policy distillation by "visual advantage," the teacher's log-prob gain from fine-grained image detail, so supervision targets vision-critical tokens.
- [GRAFT: Graph-Tokenized LLMs for Tool Planning](https://arxiv.org/abs/2605.11706) *(2026)* — On-policy tool-context distillation where a subtask-privileged same-model teacher supervises the student's own tool-token trajectories, curing exposure bias in graph-tokenized tool planning.
- [Adversarial Dual On-Policy Distillation from Expressive Flow-based Teacher](https://arxiv.org/abs/2605.27095) *(2026)* — Co-trains a demonstration-learned flow-matching teacher that supplies reward and action signals on embodied student rollouts, enabling OPD without a fixed strong teacher.
- [Data-Efficient On-Policy Distillation for Automatic Speech Recognition](https://arxiv.org/abs/2605.28139) *(2026)* — Cross-modal OPD for speech recognition: a compact audio-conditioned student learns from a frozen ASR teacher scoring its own transcript rollouts.
- [GAPD: Gold-Action Policy Distillation for Agentic Reinforcement Learning in Knowledge Base Question Answering](https://arxiv.org/abs/2605.29584) *(2026)* — Agentic-KBQA OPD distilling a gold-action-conditioned self-teacher onto entity-anchored student action spans, densifying sparse outcome rewards.
- [GDSD: Reinforcement Learning as Guided Denoiser Self-Distillation for Diffusion Language Models](https://arxiv.org/abs/2605.29398) *(2026)* — Diffusion-LLM RL-as-self-distillation matching denoiser logits to an advantage-guided self-teacher, bypassing ELBO surrogate likelihood bias.
- [CollectionLoRA: Collecting 50 Effects in 1 LoRA via Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2605.25378) *(2026)* — DMD-based multi-teacher distillation consolidating many effect LoRAs into one student LoRA for few-step image editing.
- [Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding](https://arxiv.org/abs/2606.00564) *(2026)* — Decomposes vision-language distillation into language-prior and visual-grounding gradients, steering student-rollout updates toward the visual subspace to fix perceptual bottlenecks.

### Speculative Decoding (Draft-Model Training)

Draft-model training for speculative decoding shares OPD's core loop: the draft (student) generates, the target (teacher) verifies, and the draft is updated to match. Included for breadth even though the goal is inference acceleration rather than student capability.

- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177) *(2023)* — Continuously updates draft on observed queries via KD; 1.42×-2.17× latency gains.
- [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461) *(2023)* — Aligns draft with target via on-policy data and task-tailored divergence (ICLR 2024).
- [HASS: Learning Harmonized Representations for Speculative Sampling](https://arxiv.org/abs/2408.15766) *(2024)* — Harmonized objective and context distillation fixes train-decoding inconsistency.
- [Falcon: Faster and Parallel Inference through Enhanced Semi-Autoregressive Drafting](https://arxiv.org/abs/2412.12639) *(2024)* — Coupled Sequential Glancing Distillation strengthens inter-token dependencies in semi-AR drafters.
- [CORAL: Consistent Representations across Multi-step Training with Lighter Speculative Drafter](https://arxiv.org/abs/2502.16880) *(2025)* — Cross-step representation alignment for multi-step drafter training (ACL 2025).
- [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) *(2025)* — Direct token prediction with multi-layer feature fusion under on-policy training-time test; up to 6.5×.
- [MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of VLMs](https://arxiv.org/abs/2505.10526) *(2025)* — Adapts SLM into VLM drafter via self-distilled visual instruction tuning.
- [DVI: Draft, Verify, and Improve — Toward Training-Aware Speculative Decoding](https://arxiv.org/abs/2510.05421) *(2025)* — Self-speculative drafter trained online from verifier decisions via KL→RL schedule.
- [ReSpec: Optimizing Speculative Decoding in Reinforcement Learning Systems](https://arxiv.org/abs/2510.26475) *(2025)* — Evolves drafter during RL via reward-weighted distillation on rollouts.
- [DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting](https://openreview.net/forum?id=CRgWv0kWjF) *(2026)* — Multimodal speculative-reasoning drafter with verifier-gated parallel execution.
- [MSD: Speculative Decoding Reimagined for Multimodal Large Language Models](https://arxiv.org/abs/2505.14260) *(2025)* — Decouples text/visual tokens in draft; two-stage training lifts MLLM speedups to 2.29–2.46×.
- [SpecVLM: Fast Speculative Decoding in Vision-Language Models](https://arxiv.org/abs/2509.11815) *(2025)* — Elastic visual compressor plus online-logit distillation; 2.5–2.9× end-to-end VLM speedups.
- [ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](https://arxiv.org/abs/2509.15235) *(2025)* — Lightweight vision adaptor compresses image tokens; trained on target-generated long responses.
- [Aurora: When RL Meets Adaptive Speculative Training](https://arxiv.org/abs/2602.06932) *(2026)* — Online continual draft training; target verifications stream into FKL/RKL fine-tuning then hot-swap into serving.
- [SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting](https://arxiv.org/abs/2605.07243) *(2026)* — Block-iterative drafter with layer-wise shift; valid-prefix masking and cost-aware bandit adaptation.
- [SFDD: Flatter Tokens are More Valuable for Speculative Draft Model Training](https://arxiv.org/abs/2601.18902) *(2026)* — Sample-level flatness filters EAGLE training data; 2× speedup at 50% data with <4% inference-speedup loss.
- [OmniDraft: A Cross-vocabulary, Online Adaptive Drafter for On-device Speculative Decoding](https://arxiv.org/abs/2507.02659) *(2025)* — Online on-policy distillation on the draft's own generated tokens; cross-vocabulary n-gram cache lets one drafter serve any target.
- [Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient LLMs](https://arxiv.org/abs/2605.27255) *(2026)* — Trains an OPD-aligned confidence head whose acceptance decision replaces speculative decoding's verifier pass, unifying latent input compression with multi-token-prediction output.
- [LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](https://arxiv.org/abs/2602.23881) *(2026)* — Replaces KL-proxy draft training with objectives directly targeting acceptance rate, since capacity-limited drafters minimizing KL converge to low-acceptance solutions.
- [Draft-OPD: On-Policy Distillation for Speculative Draft Models](https://arxiv.org/abs/2605.29343) *(2026)* — Draft-model OPD that replays drafting from verification-exposed error positions, training on target feedback over both accepted and rejected proposals.
- [Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding](https://arxiv.org/abs/2509.22134) *(2025)* — Trains the draft model on its own verified tree rollouts via a group-standardized acceptance-length reward, aligning training with tree-based decoding.

## Technical Reports and Industrial Recipes

Production training pipelines that use OPD as a post-training stage.

| Year | System | OPD Usage | Link |
|------|--------|-----------|------|
| 2024 | Gemma 2 | KD as alternative to next-token prediction for 2B and 9B students | [arXiv](https://arxiv.org/abs/2408.00118) |
| 2025 | Qwen3 | Strong-to-weak; off-policy then on-policy distillation | [arXiv](https://arxiv.org/abs/2505.09388) |
| 2025 | Qwen3-Omni | Off-policy then on-policy distillation before GSPO | [arXiv](https://arxiv.org/abs/2509.17765) |
| 2025 | GLM-4.5 / 4.6 | Multi-stage post-training with expert model iteration and RL | [arXiv](https://arxiv.org/abs/2508.06471) |
| 2025 | HY-MT1.5 | Multi-stage translation: SFT + OPD + RL | [arXiv](https://arxiv.org/abs/2512.24092) |
| 2026 | MiMo-V2-Flash | Multi-Teacher OPD (MOPD) as post-training stage | [arXiv](https://arxiv.org/abs/2601.02780) |
| 2026 | GLM-5 | On-policy cross-stage distillation to recover earlier skills | [arXiv](https://arxiv.org/abs/2602.15763) |
| 2026 | Typhoon-S | Minimal sovereign recipe: SFT + OPD + small-scale RFT | [arXiv](https://arxiv.org/abs/2601.18129) |
| 2026 | Nemotron-Cascade 2 | Cascade RL + multi-domain on-policy distillation | [arXiv](https://arxiv.org/abs/2603.19220) |
| 2026 | Baichuan-M3 | Task RL → offline policy distillation → multi-teacher OPD | [arXiv](https://arxiv.org/abs/2602.06570) |
| 2026 | MobileLLM-R1.5 | Final-stage on-policy KD as primary improvement over R1 | [model card](https://huggingface.co/facebook/MobileLLM-R1.5-950M) |
| 2026 | Nanbeige4-3B-Thinking | OPD preferred over off-policy for math reasoning | [model card](https://huggingface.co/Nanbeige/Nanbeige4-3B-Thinking-2510) |
| 2026 | DeepSeek-V4 | Domain-expert SFT+GRPO → unified model consolidation via OPD | [report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |
| 2026 | Qwen3.5-Omni | Specialist distillation → privileged-input self-distillation aligning audio to text | [arXiv](https://arxiv.org/abs/2604.15804) |
| 2026 | HY-Embodied-0.5 | 32B → 2B on-policy distillation; student rollouts, teacher token-level supervision | [arXiv](https://arxiv.org/abs/2604.07430) |
| 2026 | KAT-Coder-V2 | Specialize-then-Unify: 5 domain-expert agents → unified via OPD on student trajectories | [arXiv](https://arxiv.org/abs/2603.27703) |
| 2026 | Cursor Composer 2.5 | Hint-conditioned self-teacher OPD KL added to RL for targeted behaviors (tool calls, style); built on Kimi K2.5 | [blog](https://cursor.com/blog/composer-2-5) |

## Frameworks and Implementations

### Training Frameworks

| Framework | Description | Link |
|---|---|---|
| TRL | GKD, GOLD, and MiniLLM trainers; most accessible starting point | [docs](https://huggingface.co/docs/trl) |
| NeMo-RL | Multi-teacher and cross-tokenizer OPD at scale | [docs](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html), [repo](https://github.com/NVIDIA-NeMo/RL) |
| veRL | Async on-policy KD trading strict on-policy guarantees for throughput | [docs](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html) |
| MS-Swift | GKD and OPSD sections in the ModelScope ecosystem | [docs](https://swift.readthedocs.io/en/latest/) |
| EasyDistill | Comprehensive KD toolkit for black-box and white-box LLM distillation | [arXiv](https://arxiv.org/abs/2505.20888) |
| KDFlow | Off-policy, on-policy, and cross-tokenizer distillation via decoupled backends | [arXiv](https://arxiv.org/abs/2603.01875), [repo](https://github.com/songmzhang/KDFlow) |
| slime | Unified RL stack supporting on-policy distillation and hindsight hints | [repo](https://github.com/THUDM/slime) |
| OpenClaw-RL | Agentic RL stack with hindsight-guided OPD | [arXiv](https://arxiv.org/abs/2603.10165) |
| NexRL | Dedicated on-policy distillation recipes | [repo](https://github.com/nex-agi/NexRL) |
| SkyRL | OPD examples and blog resources | [repo](https://github.com/NovaSky-AI/SkyRL) |
| ATLAS | Continual-learning framework using GKD/GRPO from runtime traces | [docs](https://docs.arc.computer/introduction) |
| AReaL | OPD and KDRL over student-sampled trajectories with teacher log-prob guidance | [docs](https://github.com/inclusionAI/AReaL/blob/main/docs/en/algorithms/distillation.md) |
| rLLM | Agent RL framework (UC Berkeley Sky) with first-class OPD: `examples/math_distill/` (DeepMath OPSD + `train_deepmath_distill_tinker.{py,sh}`) and `rllm/trainer/distill/` modules over verl or tinker backends | [docs](https://docs.rllm-project.com/), [repo](https://github.com/rllm-org/rllm) |
| SpecForge | Speculative draft training with EAGLE-3 support and hybrid parallelism | [arXiv](https://arxiv.org/abs/2603.18567), [repo](https://github.com/sgl-project/SpecForge) |
| TorchSpec | Torch-native speculative draft training with disaggregated inference/training; streams target hidden states via Mooncake store; Kimi-K2.5/MiniMax-M2.5/Qwen3-Coder-Next examples | [blog](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/), [repo](https://github.com/lightseekorg/TorchSpec) |
| Tinker Cookbook | Thinking Machines' Tinker SDK recipes for off-policy KD, single/multi-teacher OPD, multi-turn tool use | [recipes](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation), [repo](https://github.com/thinking-machines-lab/tinker-cookbook) |
| ROLL | Alibaba's scalable RL library for LLMs/VLMs with an OPD pipeline | [repo](https://github.com/alibaba/ROLL) |

### Implementations

- [OPSD](https://github.com/siyan-zhao/OPSD) — Official code for Self-Distilled Reasoner / OPSD.
- [SCOPE](https://github.com/machine981/SCOPE) — Dual-path OPD: student-PPL-weighted MLE for correct rollouts, teacher-PPL-weighted KL for incorrect.
- [CaOPD](https://github.com/SalesforceAIResearch/CaOPD) — K student rollouts → empirical success rate → confidence target replacement → reverse-KL OPD.
- [OPSD-OnPolicyDistillation](https://github.com/HJSang/OPSD_OnPolicyDistillation) — verl-based OPD with separate teacher, agent-loop rollouts, and memory-efficient execution.
- [nano-opd](https://github.com/Athe-kunal/nano-opd) — Hackable OPD library decoupling vLLM rollout, FSDP training, and teacher forwards across independent GPU groups.
- [Rethinking OPD](https://github.com/thunlp/OPD) — Official code for Rethinking OPD, with verl-based scripts and top-k teacher–student overlap diagnostics merged upstream.
- [DiffusionOPD](https://github.com/ali-vilab/DiffusionOPD) — Official implementation of round-robin multi-task diffusion OPD distilling task-specialized teachers into one student along its own rollout trajectories.

## Acknowledgments

This list draws on the parallel curation effort at [thinkwee/AwesomeOPD](https://github.com/thinkwee/AwesomeOPD), which provided pointers to several papers (notably speculative-decoding draft training, BoN distillation, self-play, multilingual and crosslingual self-distillation, clinical and multimodal agentic OPD, additional industrial reports, and several training frameworks). The two lists organize differently — thinkwee/AwesomeOPD groups by feedback signal and access mode; this list groups by methodological role — and are best read together.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for criteria, section placement, and formatting.

- **Inclusion criteria:** the work should involve student rollouts as central to the learning signal, or directly enable OPD deployment (cross-tokenizer, frameworks, etc.).
- **Entry format:** `[Title](url) *(Year)* — One-line description.` See [CONTRIBUTING.md](CONTRIBUTING.md) for full examples.

## Citation

```bibtex
@software{awesome-on-policy-distillation,
  title = {{Awesome On-Policy Distillation}},
  author = {Liu, Chris Yuhao and others},
  year = {2026},
  doi = {10.5281/zenodo.19411493},
  url = {https://github.com/chrisliu298/awesome-on-policy-distillation},
  version = {v1.0.0}
}
```

---
