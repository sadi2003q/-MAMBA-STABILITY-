# Does a Built-In Stability Guarantee Make a Learned Controller Safe?
## Research Proposal and Project Brief — Stability Certificates of Selective State-Space Models (Mamba) in Closed-Loop Model Predictive Control

**Author:** Md. Adnan Abdullah Sadi — Department of Electrical and Computer Engineering, North South University, Dhaka, Bangladesh
**Document date:** 9 October 2026
**Status:** Project restart. This document is the starting brief for a fresh round of work.

---

## How to read this document

This document is written for two audiences at once:

- **A non-technical reader** (or a newcomer to the project) should read Sections 1, 2, 4, 7 and 9. These explain the problem in plain language with analogies and no mathematics.
- **A technical reader** (a supervisor, a collaborator, or an AI assistant helping with the work) should read everything, especially Section 3 (mathematical formulation), Section 6 (hypotheses), Section 7 (what success looks like, precisely) and Section 8 (evidence from the first attempt).

Every technical term is defined before it is used. A glossary is at the end (Section 11).

**Note to an AI assistant receiving this document:** treat Section 8 as prior evidence, not as settled conclusions to defend. The project is restarting, and the goal is to answer the research question honestly — a well-supported negative answer is an acceptable outcome. Citations in Section 5 were collected up to October 2026; verify any detail before quoting it in a paper.

---

## 1. The idea in one paragraph

Mamba is a modern type of neural network that reads data one step at a time while carrying an internal running memory. A 2025 paper proved that Mamba's internal memory can never grow out of control — it is mathematically guaranteed to be stable, no matter what data the network was trained on. A separate 2026 paper showed that Mamba works very well as the "prediction engine" inside an automatic controller for physical machines. Nobody has checked whether the first result explains or protects the second. This project asks: **when a Mamba network is used to predict and control a physical system, does its built-in stability guarantee actually make its predictions — and therefore the controller — stable?** If yes, engineers could certify learned controllers before deploying them. If no, we need to know why, and what kind of guarantee would actually work.

---

## 2. The problem in plain language

### 2.1 An everyday analogy: the safe dial and the unpredictable heater

Imagine a thermostat whose dial is physically locked between 0 and 40 degrees. That is a real, hardware-enforced guarantee: the dial can never go out of range.

Now connect that thermostat to an unusual heater that reacts to the dial's position in a wild, unpredictable way — a tiny change of the dial near 20 degrees might send the room from freezing to boiling. The dial never breaks its guarantee. But the **room temperature** — the thing you actually care about — can still swing out of control.

> **A guarantee about the internal mechanism is not automatically a guarantee about the output you care about.**

In this project:
- **the dial** is Mamba's internal memory (its *hidden state*);
- **the guaranteed range** is the mathematical stability proof;
- **the heater** is the part of the network that turns internal memory into a prediction about the physical world;
- **the room temperature** is the physical system being controlled — a robot arm, a drone, a water tank, or (in our experiments) an oscillating circuit.

### 2.2 Background ideas, explained simply

**A sequence model with memory.** Some neural networks read data one time-step at a time and keep a running summary — a memory — that is updated at each step: *new memory = (old memory, partly faded) + (new information)*. Older examples are recurrent neural networks and the Long Short-Term Memory network (LSTM). Mamba is a newer and faster member of this family.

**Stability.** A memory is *stable* if it cannot grow forever. Picture a snowball rolling downhill: if it can only shrink or stay the same size, it is stable; if it can grow without limit, it is unstable and eventually turns into nonsense numbers.

**Mamba's "selective" design.** Older memory models used one fixed update rule at every step. Mamba recomputes three small "gates" at every step from the current input: how much new information to let in, how much of the memory to show in the output, and how big a time-step to take. This flexibility is what makes Mamba powerful. Crucially, **the stability proof only covers the part of the update that controls how fast old memory fades. It says nothing about the other gates.**

**Model predictive control.** A strategy for controlling machines. At every moment the controller uses a prediction model to *imagine* several possible action sequences a few seconds into the future, picks the sequence whose imagined future looks best, carries out only the first action, and then repeats the whole process a moment later. It is like a chess player thinking a few moves ahead before every move. The controller can only be as good as its prediction model.

**One-step prediction versus chained prediction (rollout).** There are two ways to use a prediction model:
- *One-step:* give the model the true current state, ask it for the next state. Every input is correct.
- *Chained (also called rollout):* give it the true state once, then feed it **its own prediction** as the next input, again and again. Small errors can pile up — step 2 is built on a slightly wrong step 1, step 3 on a slightly wrong step 2, and so on.

The controller described above uses chained prediction for its whole planning window. So **what matters for control is chained-prediction accuracy, not one-step accuracy.**

**Teacher forcing.** The most common way to train a sequence model: always feed it the true previous value during training. It is fast and easy, but the model never practices recovering from its own mistakes, so it can be excellent at one-step prediction and poor at chained prediction.

### 2.3 Why this matters in the real world

Learned prediction models are increasingly used to control physical machines where failure is costly: drones, vehicles, chemical processes, power electronics, medical devices. For these, "we tested it and it seemed fine" is not enough — engineers want a guarantee *before* deployment. A neural network whose stability is proven by its very design is therefore extremely attractive. But if the proof covers the wrong quantity, it can create **false confidence**: a system that looks certified on paper but whose predictions still drift off course when used for control.

---

## 3. Mathematical formulation (for technical readers)

### 3.1 The physical system (the "plant")

We consider a discrete-time nonlinear system with state $x_t \in \mathbb{R}^{n_x}$ and control input $u_t \in \mathbb{R}^{n_u}$:

$$x_{t+1} = f(x_t, u_t).$$

Our main benchmark is the controlled Van der Pol oscillator, discretised with forward Euler and step $\Delta t$:

$$\dot x_1 = x_2, \qquad \dot x_2 = \mu (1 - x_1^2)\,x_2 + u,$$

with $\mu = 1$, which produces a **limit cycle** (a closed loop that trajectories settle onto). This is the same benchmark used by Cevaal et al. (2026), so results are directly comparable.

### 3.2 The learned model: a selective state-space model

The Mamba core keeps a hidden state $h_t \in \mathbb{R}^{D \times N}$ ($D$ channels, $N$ memory entries per channel). At each step, from the current (embedded) input $s_t$, it computes input-dependent quantities:

$$\Delta_t = \operatorname{softplus}(W_\Delta s_t + b_\Delta) > 0, \qquad B_t = W_B s_t, \qquad C_t = W_C s_t,$$

and a fixed decay parameter constrained to be negative, $A = -\exp(A_{\log}) < 0$. The discretised update is

$$\bar A_t = \exp(\Delta_t \odot A), \qquad h_t = \bar A_t \odot h_{t-1} + (\Delta_t \odot B_t)\, s_t, \qquad y_t = C_t^\top h_t + D\, s_t,$$

where $\odot$ is element-wise multiplication. When used as a dynamics model, the network output is turned into a physical prediction, typically through a residual readout

$$\hat x_{t+1} = \hat x_t + g_\theta(y_t),$$

where $g_\theta$ is a learned output map (in the full architecture: normalisation, gating and a linear projection).

### 3.3 The certificate (Halloran et al., 2025)

Because $\Delta_t > 0$ and $A < 0$ element-wise,

$$0 < \bar A_t = \exp(\Delta_t \odot A) < 1 \quad \text{for all } t \text{ and all inputs}.$$

Hence the map $h_{t-1} \mapsto h_t$ (holding inputs fixed) is a contraction in the hidden state: the maximal Lyapunov exponent of the latent recurrence is non-positive. This holds **by construction**, for any trained weights and any data. We call this the **latent certificate**.

### 3.4 What the controller actually uses

Model predictive control solves, at every time $k$,

$$\min_{u_{0:N-1}} \; \sum_{i=0}^{N-1} \ell(\hat x_i, u_i) + \ell_N(\hat x_N) \quad \text{s.t.}\quad \hat x_{i+1} = F_\theta(\hat x_i, u_i, h_i),\;\; \hat x_0 = x_k,\;\; u_i \in \mathcal U,$$

where $F_\theta$ is the learned model **applied to its own predictions** (chained prediction). The quality of the plan depends on the rollout error

$$e_i = \hat x_i - x_i.$$

### 3.5 The gap: latent stability is not prediction-error stability

Linearising around the true trajectory, the rollout error obeys

$$e_{t+1} \approx J_t\, e_t + \varepsilon_t, \qquad J_t = \frac{\partial \hat x_{t+1}}{\partial \hat x_t},$$

where $\varepsilon_t$ is the one-step model error. Unrolling,

$$\|e_T\| \;\le\; \sum_{k=0}^{T-1} \Big(\prod_{j=k+1}^{T-1} \|J_j\|\Big)\, \|\varepsilon_k\|.$$

If $\sup_t \|J_t\| \le \gamma < 1$, rollout error stays bounded by $\varepsilon_{\max}\,\frac{1-\gamma^T}{1-\gamma}$. If $\|J_t\| > 1$ often, it can grow geometrically.

The crucial point: under self-feeding, the hidden state and the physical prediction influence each other, so the relevant Jacobian is that of the **augmented state** $z_t = (\hat x_t, h_t)$:

$$\mathcal J_t = \begin{pmatrix} \partial \hat x_{t+1}/\partial \hat x_t & \partial \hat x_{t+1}/\partial h_t \\ \partial h_{t+1}/\partial \hat x_t & \partial h_{t+1}/\partial h_t \end{pmatrix}.$$

The latent certificate controls (part of) the bottom-right block only. The off-diagonal blocks — which depend on the readout $C_t$, the input gate $B_t$, the step size $\Delta_t$'s dependence on the fed-back prediction, and the residual readout $g_\theta$ — are **completely unconstrained**.

**A two-line counterexample.** Take the scalar augmented Jacobian

$$\mathcal J = \begin{pmatrix} 1 & 1 \\ 1 & 0.5 \end{pmatrix}.$$

The latent block satisfies the certificate ($0.5 < 1$), yet the eigenvalues are $\tfrac{1.5 \pm \sqrt{4.25}}{2} \approx \{1.78,\, -0.28\}$, so errors grow by roughly a factor of 1.78 per step. **The latent certificate holds, and rollout still diverges.** A formal version of this argument is one of the project's intended contributions.

### 3.6 A deeper structural reason (Chung, Choi and Kim, 2026)

Conditional on the input sequence, the Mamba recurrence is **affine** in the hidden state: $h_t = \bar A_t h_{t-1} + b_t$. Chung, Choi and Kim (May 2026) prove that any affine recurrence which exactly preserves a set of distinguishable internal states must act as the identity on the subspace separating those states — so any error injected into that subspace is carried forward unchanged and is **never corrected**. They show that only **state-dependent** update maps can escape this obstruction and actively contract drift.

Their result is proved for symbolic state tracking, not continuous control. Whether it extends to continuous dynamical systems is an open question and a candidate contribution of this project. It suggests that penalties applied on top of an unmodified affine recurrence can only make the memory *decay faster*; actively *correcting* error requires changing the recurrence itself.

### 3.7 A trap any fix must avoid: limit cycles

The Van der Pol limit cycle is locally *expanding* in some regions: the true system's Jacobian has largest singular value above 1 along parts of the orbit. A model forced to be globally contracting ($\|J_t\| < 1$ everywhere) **cannot represent a limit cycle at all** — it would collapse every trajectory to a single point and destroy one-step accuracy.

Any proposed physical-space certificate must therefore bound **error dynamics**, not state dynamics. Suitable notions include:
- *incremental stability* and *contraction analysis* (Lohmiller and Slotine, 1998): neighbouring trajectories converge in a possibly state-dependent metric;
- *transverse contraction* (Manchester and Slotine): contraction only in directions transverse to the flow, which is exactly the property of an orbitally stable limit cycle;
- *relative bounds*: the model's error amplification must not exceed the true system's error amplification by more than a margin.

---

## 4. Why this question exists — the gap between two research communities

| | Halloran et al. (2025) | Cevaal et al. (2026) |
|---|---|---|
| **Studies** | Mamba's internal memory | Mamba as a predictor inside a controller |
| **Shows** | Memory is provably stable (formal proof) | Mamba-based control works and is fast (experiments) |
| **Tested on** | Large language models; fine-tuning robustness | Van der Pol oscillator, four-tank system, Quanser Aero2 hardware |
| **Mentions the other's topic?** | No control setting | No stability certificate; lists "closed-loop stability analysis framework for Mamba-based predictive control" as future work |

One paper has the **guarantee** but no **control setting**. The other has the **control setting** but no **guarantee**. They come from different communities (machine-learning theory versus control engineering) and cite largely different literature. The question connecting them — *does the guarantee explain or protect the control performance?* — has not been asked. This is the gap this project fills.

There is also a subtle reason the gap is easy to overlook: Cevaal et al.'s predictor emits the **whole prediction horizon in one forward pass** (a "direct multi-step" design), so it never feeds its own predictions back in, and errors cannot compound in the usual way. Their introduction explicitly motivates this design by the error accumulation of one-step predictors. Many other learned controllers, however, use one-step models applied repeatedly, and direct multi-step models have their own limits (a fixed horizon baked into the architecture). Both regimes must be studied.

---

## 5. Related and recent work

Collected up to October 2026. Verify every detail before citing.

### 5.1 Directly on the research question

| Work | What it contributes | What it leaves open |
|---|---|---|
| **Gu and Dao (2023)**, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv:2312.00752 | The architecture | No stability or control analysis |
| **Halloran, Gulati and Roysdon (2025)**, "Mamba State-Space Models Are Lyapunov-Stable Learners", TMLR, arXiv:2406.00209 | The latent certificate (Section 3.3) | Never tested in control; says nothing about gates or readout |
| **Cevaal, de Jong and Lazar (2026)**, "Mamba Sequence Modeling meets Model Predictive Control", arXiv:2604.13857 | Full Mamba as direct multi-step predictor in model predictive control; beats LSTM on speed and often accuracy; hardware demo. Van der Pol model ≈3,400 parameters | No stability theory; flags closed-loop stability analysis as future work |
| **Cao et al. (2026)**, "A Control-Theoretic View of Mamba on Stability and Robustness", ICML 2026 (PMLR v306) | Bounded-input bounded-output stability and a linear error-growth bound for a **single** Mamba layer | States that the multi-layer, full-network case remains open |
| **Zubić and Scaramuzza (2025)**, "Regularity and Stability Properties of Selective SSMs with Discontinuous Gating", arXiv:2505.11602 | Passivity / linear-matrix-inequality certificate at the core level | Core certificate reduced internal violations but **did not improve end-task error** — a close parallel to this project's question, in forecasting rather than control |
| **Chung, Choi and Kim (2026)**, "Rethinking State Tracking in Recurrent Models Through Error Control Dynamics", arXiv:2605.07755 | Proof that affine recurrences cannot correct drift in preserved directions; state-dependent maps can (Section 3.6) | Symbolic tracking only; continuous control untested |
| **Biswas (2026)**, "A Comparative Study of Accuracy and Rollout Stability of Temporal Surrogate Models", arXiv:2605.24868 | Five model types on chaotic systems under matched training; **an LSTM with a contractive one-step Jacobian still showed large error growth in chained rollout**; short-horizon accuracy did not predict stability | No state-space models or Mamba; no control |

### 5.2 Stable learned dynamics and certified control

- **Kolter and Manek (2019)**, "Learning Stable Deep Dynamics Models", NeurIPS — stability by construction via a learned Lyapunov function; globally stable, so cannot represent limit cycles directly.
- **Lohmiller and Slotine (1998)**, "On Contraction Analysis for Non-linear Systems", Automatica — the foundation for incremental stability.
- **Singh, Richards, Sindhwani, Slotine and Pavone (2018)**, "Learning Stabilizable Dynamical Systems via Control Contraction Metrics", arXiv:1808.00113.
- **Gallieri et al. (2020)**, "Neural Lyapunov Model Predictive Control", arXiv:2002.10451 — learned Lyapunov function as a terminal ingredient for model predictive control.
- **Li, Han and Yin (2025)**, "MamKO: Mamba-based Koopman Operator for Modeling and Predictive Control", ICLR 2025 — uses Mamba to generate a linear time-varying model rather than as a full predictor.
- **Hu et al. (2024)**, "State-Space Models are Accurate and Efficient Neural Operators for Dynamical Systems", arXiv:2409.03231.
- **"Sparse Mamba"** (2024), arXiv:2409.00563 — imposing controllability, observability and stability structure on state-space models.

### 5.3 Compounding error and training for rollout

- **Bengio et al. (2015)**, "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" — the classic fix for teacher-forcing exposure bias.
- **Lambert, Pister and Calandra (2022)**, "Investigating Compounding Prediction Errors in Learned Dynamics Models", arXiv:2203.09637.

---

## 6. Research questions and hypotheses

**Main question.** Does the latent stability certificate of a selective state-space model transfer to stability of the physical predictions used in closed-loop control? If not, what additional condition is needed, and can it be built into the architecture?

| Code | Hypothesis | How it is tested |
|---|---|---|
| **H1** | The latent certificate is *binding*: removing it changes chained-prediction error | Train identical models with $A$ constrained negative versus unconstrained (free sign), many seeds; compare rollout error |
| **H2** | Latent stability does not bound physical rollout error | Show models with the certificate satisfied but rollout error growing; formal counterexample (Section 3.5) |
| **H3** | Rollout-aware training matters more than the certificate | Teacher forcing versus training on chained predictions, with and without the certificate |
| **H4** | A **physical-space / error-dynamics certificate** (Section 3.7) reduces rollout error | Proposed architecture versus unmodified baseline |
| **H5** | The benefit comes from the *mechanism*, not extra parameters | Proposed method versus a parameter-matched control (for example, a fixed, state-independent version of the same correction) |
| **H6** | The improvement carries through to the closed loop | Model predictive control tracking and stabilisation under disturbance and parameter drift |
| **H7** | The effect is general, not specific to Mamba | Same method applied to an LSTM baseline |

Each hypothesis has an explicit, falsifiable pass/fail criterion declared **before** running the experiment.

---

## 7. What a positive outcome would look like

### 7.1 In plain language

A positive outcome means: we design a new kind of guarantee — one that covers **the predictions the controller actually uses**, not just the network's internal memory — and we show, both by proof and by experiment, that a Mamba network carrying this guarantee:

1. makes chained predictions whose errors stay small instead of piling up;
2. still predicts the physical system accurately one step at a time (the guarantee does not "cost" accuracy);
3. still captures the system's natural behaviour, such as oscillating forever around a limit cycle;
4. produces a controller that is measurably more reliable — especially when the machine is pushed or its physical properties change;
5. does so because of the specific design idea, not merely because the network got bigger.

The headline sentence of a positive paper would be:

> *"The built-in latent stability guarantee of selective state-space models does not transfer to closed-loop control, because it leaves the prediction pathway unconstrained. We prove this, identify the missing condition, and propose an architecture whose certificate bounds chained-prediction error directly. It reduces rollout error by X% across Y systems while preserving one-step accuracy and limit-cycle behaviour, and yields a provably stable predictive controller."*

### 7.2 In precise terms — the success criteria

A result counts as **positive** only if all of the following hold:

| # | Criterion | Threshold |
|---|---|---|
| 1 | **Theorem.** A provable bound on chained-prediction error for the proposed architecture, e.g. $\|e_T\| \le \varepsilon_{\max}\frac{1-\gamma^T}{1-\gamma}$ in a suitable (possibly state-dependent) metric, *plus* a counterexample showing the latent certificate alone gives no such bound | Formal statement and proof |
| 2 | **Rollout error** lower than the unmodified baseline | Median rollout root-mean-square error lower; Mann–Whitney test p < 0.05 **and** Cliff's delta effect size > 0.33; at least 8 seeds |
| 3 | **Mechanism, not capacity** | Proposed method beats a parameter-matched control version (same criteria as row 2) |
| 4 | **No accuracy cost** | One-step error within about 10% of baseline |
| 5 | **Dynamics preserved** | Limit cycle reproduced (period and amplitude within a few percent); zero divergent runs |
| 6 | **Generality across systems** | Holds on at least 2–3 nonlinear systems (for example Van der Pol, a pendulum or cart-pole, the four-tank system) |
| 7 | **Closed loop** | Better tracking or stabilisation than unconstrained Mamba-based control and LSTM-based control, under disturbances and parameter drift, at matched parameter count |
| 8 | **Both prediction regimes** | Reported for chained (one-step-repeated) and direct multi-step prediction |

### 7.3 An illustrative positive results table

*Example only — these numbers are invented to show the shape of a convincing result.*

| Model | Training | One-step error | Rollout error (median, 50 steps) | 95% confidence interval | Divergent runs |
|---|---|---|---|---|---|
| Baseline Mamba | teacher forcing | 0.007 | 0.310 | [0.28, 0.34] | 0/10 |
| **Proposed** | teacher forcing | 0.008 | **0.095** | [0.08, 0.11] | 0/10 |
| Baseline Mamba | rollout training | 0.018 | 0.077 | [0.07, 0.08] | 0/10 |
| Control (fixed correction) | rollout training | 0.019 | 0.074 | [0.07, 0.08] | 0/10 |
| **Proposed** | rollout training | 0.019 | **0.041** | [0.03, 0.05] | 0/10 |

What makes it convincing: the proposed rows are clearly lower than **both** the baseline **and** the parameter-matched control; confidence intervals do not overlap; one-step error barely changes; nothing diverges.

### 7.4 What a negative outcome would look like — and why it is still publishable

If no tested certificate rescues chained-prediction stability, the paper becomes: *"Architectural stability guarantees for selective state-space models do not transfer to control; here is the formal reason, here is consistent evidence across systems and seeds, and here is what actually helps (rollout-aware training), which is architecture-agnostic."* That corrects a plausible but unverified assumption and is a legitimate contribution. The rule for this project: **never tune until something looks positive.** Hyperparameters are fixed by a pre-registered sweep; the hypothesis is then tested once, at full scale.

---

## 8. Evidence from the first attempt (September–October 2026)

The project ran eight experimental phases before this restart. All results used the Van der Pol oscillator and mostly 2–3 seeds, so they are **prior evidence and direction**, not publishable conclusions.

| Phase | Question | Finding |
|---|---|---|
| 0 | Does the certificate hold? | Yes, exactly, in every run |
| 1 | Does it keep chained prediction stable (teacher forcing)? | No — rollout error 200–300× larger than one-step error |
| 2 | Is the training method responsible? | Largely yes — training on chained predictions fixed most of it |
| 3–4 | Does the certificate help inside a controller? | No measurable difference with versus without it |
| 5 | Is Mamba necessary? | No — a plain LSTM matched it |
| 6 | Remove the certificate entirely (free-sign $A$) | No change in rollout error; unconstrained training kept $A$ negative by itself, so the certificate appears **non-binding**. A model *initialised* unstable diverged in every run |
| 7 | Full-size Mamba (≈6,000 parameters) plus physical-space penalties: bounded output gate, spectral normalisation of the readout, penalty on excess error amplification, and a combination | **No penalty beat the baseline.** Spectral normalisation hurt accuracy; the error-amplification penalty hurt under teacher forcing and only tied under rollout training. Diagnostic: the trained model's single-step Jacobian was already *below* the true system's, so error was spreading through the **hidden state**, not the readout |
| 8 | State-dependent correction inside the recurrence (motivated by Section 3.6), with a fixed-gate control | Only a short pilot run (undertrained); inconclusive. The gate was confirmed to vary with the state |

**Practical lessons for the restart:**
1. Use at least 8 seeds before reading any significance test — at 3 seeds the Mann–Whitney test cannot reach p < 0.05 regardless of effect size.
2. Report divergent runs as a divergence rate, never as a clamped error value.
3. Always include a parameter-matched control arm so improvements can be attributed to the mechanism.
4. Any certificate must respect the limit cycle (Section 3.7).
5. Penalties on top of an affine recurrence did not work; architectural changes to the recurrence are the remaining promising direction.
6. Budget compute: a full-size model trained on chained predictions took 3–9 minutes per run on a single Kaggle T4 graphics card; grids must be sized accordingly, with per-run checkpointing.
7. Verify architecture correctness up front: the step-by-step and parallel versions of the model must give identical outputs, and gradient must reach every new component.

---

## 9. Scope, method and evaluation plan

**Systems.** Van der Pol oscillator (primary, matches Cevaal et al.); at least one more nonlinear system (pendulum, cart-pole, or continuously stirred tank reactor); four-tank system if time allows.

**Models.** (a) Minimal selective state-space cell for isolating mechanisms; (b) full Mamba block — expansion, depthwise causal convolution, selective state-space core, gating, root-mean-square normalisation, residual connections, several layers — at Cevaal et al.'s parameter scale; (c) parameter-matched LSTM baseline.

**Training regimes.** Teacher forcing; training on chained predictions (full or scheduled); direct multi-step.

**Metrics.** One-step error; chained-prediction error over a 50-step horizon; error growth versus horizon; finite-horizon error amplification of model versus true system; divergence rate; limit-cycle fidelity; closed-loop tracking error and control effort.

**Statistics.** At least 8 seeds per configuration; medians with bootstrap 95% confidence intervals; Mann–Whitney tests with Cliff's delta; pass/fail criteria declared before each experiment.

**Compute.** Kaggle free graphics cards (NVIDIA T4). Experiments must checkpoint after every run and resume automatically.

**Phased plan (each phase ends with a falsifiable pass/fail check):**
1. Reproducible benchmarking and certificate ablation at full seed count.
2. Formal analysis: counterexample and theorem on why latent contraction does not bound physical error; extension of Chung et al. to continuous dynamics.
3. Design and test of an error-dynamics certificate that respects limit cycles.
4. Generalisation to more systems and to LSTM.
5. Closed-loop model predictive control with the certificate, under disturbance and parameter drift, against unconstrained Mamba-based control and LSTM-based control.

---

## 10. Expected contribution

1. **The first test** of whether a selective state-space model's architectural stability guarantee transfers to closed-loop control — answering the open direction raised by Cevaal et al. (2026).
2. **A formal explanation** of why latent contraction does not bound physical prediction error, connected to the affine-recurrence obstruction of Chung et al. (2026) and extended to continuous dynamics.
3. **Either** a new error-dynamics certificate and architecture with proven and demonstrated benefit (positive outcome), **or** a rigorous, well-evidenced negative result identifying what actually governs rollout stability (negative outcome). Both are contributions.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Mamba / selective state-space model** | A neural network that processes sequences step by step with a running memory whose update rule is recomputed from each input |
| **Hidden state / latent state** | The network's internal memory — not directly visible |
| **Physical state** | The real quantity being predicted, e.g. position and velocity |
| **Certificate** | A formal mathematical guarantee |
| **Latent certificate** | Halloran et al.'s guarantee that the hidden state cannot grow without limit |
| **Lyapunov exponent** | A number describing whether nearby trajectories separate (positive) or converge (negative) over time |
| **Contraction** | A map that brings any two points closer together |
| **Jacobian** | The matrix of how much each output changes when each input changes slightly |
| **Gates ($B_t$, $C_t$, $\Delta_t$)** | Input-dependent controls deciding what enters memory, what is read out, and the step size |
| **Model predictive control** | A controller that repeatedly plans several steps ahead with a prediction model and executes only the first step |
| **One-step prediction** | Predicting the next state from the true current state |
| **Chained prediction / rollout** | Feeding the model its own predictions repeatedly |
| **Teacher forcing** | Training by always feeding the true previous value |
| **Exposure bias** | The failure of teacher-forced models when they must use their own outputs |
| **Direct multi-step prediction** | Predicting an entire horizon in one forward pass, without feeding predictions back |
| **Limit cycle** | A closed orbit that a system settles onto and oscillates around forever |
| **Van der Pol oscillator** | A standard nonlinear test system with a limit cycle |
| **LSTM (Long Short-Term Memory)** | An older, widely used recurrent neural network with gated memory |
| **Root-mean-square error** | The square root of the average squared prediction error |
| **Mann–Whitney test** | A statistical test of whether one group of values tends to be larger than another |
| **Cliff's delta** | An effect size from −1 to 1; magnitude above 0.33 is a medium-to-large effect |
| **Seed** | The random starting point of a training run; more seeds show whether a result is reliable |
