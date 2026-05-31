# THE SOFTMAX CONVERGENCE

**On the Independent Discovery of the Universal Bounded-Attention Operator by Statistical Mechanics, Economics, Neuroscience, and Computer Science — and Seven Novel Implications That Follow**

*Eric Ren · ERI Labs · Jersey City, New Jersey · 2026 · github.com/ericrenone*

---

> "The probability of a state is proportional to the exponential of its entropy — and in thermal equilibrium, the distribution over energy states is uniquely determined by the condition that entropy is maximized subject to a fixed mean energy."
>
> — Boltzmann, L. (1877). Über die Beziehung zwischen dem zweiten Hauptsatze der mechanischen Wärmetheorie und der Wahrscheinlichkeitsrechnung. *Sitzungsberichte der Akademie der Wissenschaften*, 76, 373–435.

> "The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point. The channel has a capacity C. It is not possible to transmit at a rate exceeding C without an arbitrarily large probability of error."
>
> — Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.

> "The brain is an inference machine. It does not passively register the world. It actively predicts it, and learns from the difference between prediction and reality. Precision is the key variable: it governs which prediction errors are amplified and which are suppressed — it is, in effect, the brain's attention operator."
>
> — Friston, K.J. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.

> "The solution to the rational inattention problem, when choices are discrete, is the multinomial logit model. The logit probabilities emerge not as an assumption about preference shocks, but as the optimal response to an information cost."
>
> — Matějka, F., & McKay, A. (2015). Rational inattention to discrete choices: A new foundation for the multinomial logit model. *American Economic Review*, 105(1), 272–298.

> "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
>
> — Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

---

## I. The Convergence

In 1877, Ludwig Boltzmann derived the equilibrium distribution of a physical system in thermal contact with a heat reservoir. The derivation required only two things: the principle that isolated systems maximize entropy, and the constraint that mean energy is fixed. From these two conditions — entropy maximization under a single expectation constraint — a unique distribution follows:

$$P(i) = \frac{e^{-E_i / kT}}{Z}, \quad Z = \sum_j e^{-E_j / kT}$$

This is the Boltzmann distribution. It is the founding result of statistical mechanics, taught in every physics curriculum worldwide, universally recognized. What is not universally recognized is that its derivation is completely domain-neutral. It makes no use of energy, temperature, or thermodynamics as such. It uses only: a set of alternatives, a value over those alternatives, a cost parameter governing the sharpness of the allocation, and an entropy constraint. Any problem with these four components has the same solution.

One hundred twenty-six years after Boltzmann, Christopher Sims imported Shannon's channel capacity theorem into macroeconomics. An economic agent choosing among discrete actions cannot process the full information environment; their cognitive capacity is bounded by a mutual information constraint. Sims asked: what is the optimal allocation of this bounded capacity? The derivation required only two things: the objective that expected utility is maximized, and the constraint that information processing cost does not exceed capacity. The solution is unique:

$$P^*(a|s) \propto \exp\left(\frac{U(a,s)}{\lambda}\right)$$

The information cost parameter $\lambda$ is the shadow price of channel capacity. The formula is the softmax. It is formally identical to the Boltzmann distribution with utility $U$ replacing negative energy $-E$ and $\lambda$ replacing $kT$. Neither Sims nor Matějka and McKay, who formalized this as the RI-Logit result in 2015, identified the Boltzmann derivation as a prior instance of the same theorem.

Also in 2010 — the same year Sims' framework was being formalized in the economics literature — Karl Friston published the most comprehensive account of neural function ever proposed: the free-energy principle. The brain minimizes variational free energy, the gap between its internal model and the sensory environment. Learning is driven by precision-weighted prediction errors: signals are amplified by a precision matrix $\Sigma$ whose entries constitute an attention operator over sensory channels. The optimal policy over actions in this framework takes the form:

$$\pi^*(a) \propto \exp\left(\frac{-\mathcal{F}(a)}{\tau}\right)$$

where $\mathcal{F}(a)$ is the expected free energy under action $a$ and $\tau = 1/\Sigma$ is the inverse precision. This is again the softmax — derived from first principles of Bayesian inference and free energy minimization, with no reference to thermodynamics or channel capacity theory. Schultz, Dayan, and Montague (1997) had already confirmed the biological substrate: dopaminergic neurons fire proportionally to prediction error $(r - V)$, not to reward $r$. The learning signal is proportional to surprise, not to value. The brain's biological update rule is the RI gradient.

In 2017, Vaswani and colleagues published an architecture in which scaled dot-product attention — implemented as $\text{softmax}(QK^T / \sqrt{d_k}) \cdot V$ — replaced every other processing mechanism and generalized to every sequence domain it encountered. The temperature parameter $\sqrt{d_k}$ was chosen for numerical stability. No connection to RI theory, thermodynamics, or free energy minimization was identified. The architecture was derived empirically.

Four derivations. Four distinct centuries and disciplines — thermodynamics (1877), information economics (2003–2015), computational neuroscience (2010), and deep learning (2017). One equation:

$$\boxed{P(i) = \frac{e^{u_i / \tau}}{\sum_j e^{u_j / \tau}}}$$

where $u_i$ is the value of alternative $i$ and $\tau$ is the temperature governing the sharpness of the allocation. Each field gave it a different name, a different physical story, and a different derivation. All four were finding the same mathematical object from different angles. That object is named here the **bounded-attention operator**.

The claim is not merely historical. It is structural: the bounded-attention operator is the **unique** solution to any optimization problem of the form *maximize expected value subject to an entropy constraint on the allocation distribution*. This characterization is domain-neutral. It applies equally to a gas in thermal equilibrium, to an economic agent with limited cognition, to a neural system minimizing predictive error, and to a language model trained by cross-entropy. The convergence is not an analogy. It is a theorem with four independent proofs.

---

## II. Timeline Genealogy: Four Independent Derivations

The four derivations share a common mathematical ancestor — Shannon's (1948) information theory — but only two of them consciously descend from it. The Boltzmann derivation predates Shannon by 71 years and is formally independent. The neuroscientific derivation (predictive coding, free energy) descends from Helmholtz (1867) and proceeds through Rao and Ballard (1999) before Friston (2010) unifies it with information theory. The machine learning derivation proceeds empirically, with no stated theoretical foundation in either Shannon or Boltzmann.

```
PHYSICS (Thermodynamic Branch, independent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1877] Boltzmann ──────────── entropy maximization ──────────────────────────────────┐
        ↓                     under ⟨E⟩ = const                                     │
[1878] Gibbs ───────────────── canonical ensemble ───────────────────────────────────┤
        ↓                     partition function Z                                   │
[Statistical mechanics ──────────────────────────────────────────────────────────┐  │
 as equilibrium theory]                                                           │  │
                                                                                  │  │
INFORMATION THEORY (Shared Ancestor)                                              │  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                  │  │
[1948] Shannon ─── channel capacity ─── entropy ─── mutual information            │  │
         │                │                                                        │  │
         ▼                ▼                                                        │  │
ECONOMICS               COMPUTER SCIENCE         NEUROSCIENCE                     │  │
━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━             │  │
[2003] Sims             [2014] Bahdanau          [1999] Rao & Ballard             │  │
 RI as capacity          Additive alignment       Predictive coding               │  │
 constraint problem      neural attention         in visual cortex                │  │
        ↓                       ↓                        ↓                        │  │
[2015] Matějka          [2017] Vaswani          [2010] Friston                    │  │
 & McKay                 Scaled dot-product       Free Energy Principle            │  │
 RI-Logit =              attention = softmax      Precision = attention            │  │
 softmax (AER)                                    operator                         │  │
        │                       │                        │                        │  │
        └───────────────────────┴────────────────────────┘                        │  │
                                │                                                  │  │
                                ▼                                                  │  │
                [2015–2017] THREE-FIELD CONVERGENCE                               │  │
                Economics, CS, Neuroscience independently                         │  │
                prove the same formula from different premises.                   │  │
                                ▼                                                  │  │
                [PRESENT] FOUR-FIELD CONVERGENCE ◄────────────────────────────────┘  │
                Boltzmann (1877) identified as the first derivation,                  │
                170 years earlier. ◄────────────────────────────────────────────────┘
```

The temporal structure of this diagram contains the most consequential datum: Boltzmann derived the solution in 1877, Matějka and McKay in 2015, Vaswani in 2017, Friston in 2010. The solution was available 126–140 years before any of the three modern derivations appeared. It was available in a discipline — thermodynamics — whose connection to information-processing optimality was not formally established until Jaynes (1957) linked maximum entropy inference to Gibbs statistics, and whose connection to cognitive science was not formalized until Friston imported the free-energy framework into neuroscience. The Boltzmann distribution has been sitting in every physics curriculum for 150 years as the solution to the optimal bounded-attention problem, without being recognized as such.

---

## III. The Mathematical Core: Four Derivations of One Equation

### 3.1 The Thermodynamic Derivation (Boltzmann, 1877)

A physical system occupies microstate $i$ with energy $E_i$. Thermal equilibrium with a reservoir at temperature $T$ is the distribution that maximizes the system's entropy subject to the constraint that mean energy equals a fixed value $\langle E \rangle$. Solving via Lagrange multipliers:

$$\max_{\{P_i\}} -\sum_i P_i \log P_i \quad \text{subject to} \quad \sum_i P_i E_i = \langle E \rangle, \quad \sum_i P_i = 1$$

yields the unique solution $P_i^* = e^{-E_i / kT} / Z$ where the Lagrange multiplier $1/kT$ is the shadow price of the energy constraint and $Z = \sum_j e^{-E_j / kT}$ is the partition function.

Replacing negative energy with utility — $u_i = -E_i$, since agents prefer high-utility options as physical systems prefer low-energy states — and $kT$ with $\tau$ (the price of the constraint, whatever its physical interpretation):

$$P^*(i) = \frac{e^{u_i / \tau}}{Z} = \text{softmax}(u / \tau)_i$$

The Boltzmann distribution is the softmax. The derivation requires no knowledge of agents, information, or decisions. It requires only entropy maximization under a single linear expectation constraint.

### 3.2 The Economic Derivation (Sims, 2003; Matějka and McKay, 2015)

An agent with utility $v_{ij}$ for action $i$ in state $j$, prior belief $\mu_j$, and channel capacity $\kappa$ solves:

$$\max_{\{P(i|j)\}} \sum_j \mu_j \sum_i P(i|j) v_{ij} - \lambda \cdot I(A; O)$$

where $I(A; O) = H(A) - \mathbb{E}_O[H(A|O)]$ is the mutual information between actions and states, and $\lambda$ is the shadow price of channel capacity. Matějka and McKay (2015) proved that the unique solution is:

$$P^*(i|j) = \frac{P_i^0 \cdot \exp(v_{ij} / \lambda)}{\sum_k P_k^0 \cdot \exp(v_{kj} / \lambda)}$$

where $P_i^0$ is an endogenous marginal prior. With uniform prior, this collapses to the pure softmax. The information cost $\lambda$ occupies exactly the structural position of the Boltzmann temperature $\tau$ — it is the shadow price of the entropy constraint.

The American Economic Review published this result in 2015. The Transformer paper appeared in 2017. Economics independently derived the mathematical foundation of the Transformer's core mechanism from first principles of bounded rationality, two years before the architecture was published.

### 3.3 The Neuroscientific Derivation (Friston, 2010; Schultz et al., 1997)

The brain maintains a generative model $Q$ that approximates the true posterior $P(s|y)$ over hidden states $s$ given observations $y$. Variational free energy is:

$$\mathcal{F} = \underbrace{D_{\text{KL}}(Q \| P(s|y))}_{\text{divergence from truth}} - \underbrace{\log P(y)}_{\text{log evidence}}$$

Minimizing $\mathcal{F}$ is equivalent to maximizing the evidence lower bound — simultaneously reducing model inaccuracy and increasing sensory evidence. The precision matrix $\Sigma$ (inverse of the prediction error covariance) governs how strongly each sensory channel's errors drive belief updating:

$$\delta = \Sigma \cdot (\hat{y} - y)$$

High precision on a channel amplifies its prediction errors; the precision matrix is an attention operator over the sensory space. The optimal policy over actions under free energy minimization assigns:

$$\pi^*(a) \propto \exp\left(-\mathcal{F}(a) / \tau\right) = \text{softmax}(-\mathcal{F} / \tau)$$

where $\tau = 1/\Sigma$ is the inverse precision. This is the softmax over negative free energy — the RI-optimal policy with the precision playing the role of $1/\lambda$.

Schultz, Dayan, and Montague (1997) provided the empirical anchor: dopaminergic neurons fire proportionally to the prediction error $(r - V)$, not to reward $r$ alone. The biological learning signal is proportional to surprise — to the gap between predicted and received value — confirming that the brain's update rule is the gradient of the mutual information objective. The dopamine signal is the RI signal.

### 3.4 The Machine Learning Derivation (Vaswani et al., 2017)

The Transformer computes attention weights via:

$$A(q_m, k_t) = \frac{\exp(q_m \cdot k_t / \sqrt{d_k})}{\sum_l \exp(q_m \cdot k_l / \sqrt{d_k})}, \quad \text{Attention}(Q, K, V) = A \cdot V$$

The inner product $q_m \cdot k_t$ measures query-key compatibility (a utility). The scaling factor $\sqrt{d_k}$ prevents gradient saturation in high-dimensional spaces (a capacity normalization). The softmax allocates attention across positions (a bounded-attention allocation). No derivation from first principles was given; the architecture was discovered empirically.

The mathematical reason it works — and the reason it generalizes to every sequence domain — is that it implements the RI-optimal policy for sequence processing, which is identically the Boltzmann equilibrium distribution over sequence positions, which is identically the Fristonian precision-weighted update. It works because it is the unique solution to the bounded-attention optimization problem instantiated in matrix algebra.

---

## IV. The Structural Alignment: Four Derivations Unified

| Dimension | Statistical Mechanics | Rational Inattention | Free Energy Principle | Transformer Attention |
|:---|:---|:---|:---|:---|
| **Origin** | Boltzmann, 1877 | Sims 2003 / Matějka-McKay 2015 | Friston 2010 | Vaswani et al. 2017 |
| **Core problem** | Equilibrium distribution under energy constraint | Action selection under capacity constraint | Belief updating under predictive error | Token relevance allocation in sequences |
| **Value function** | $-E_i$ (negative energy) | $v_{ij}$ (utility) | $-\mathcal{F}(a)$ (negative free energy) | $q_m \cdot k_t$ (dot product) |
| **Temperature / cost** | $kT$ (thermal energy) | $\lambda$ (attention shadow price) | $\tau = 1/\Sigma$ (inverse precision) | $\sqrt{d_k}$ (scaling factor) |
| **Optimal allocation** | $e^{-E_i/kT} / Z$ | $P_i^0 e^{v/\lambda} / Z$ | $e^{-\mathcal{F}(a)/\tau} / Z$ | $e^{q \cdot k / \sqrt{d_k}} / Z$ |
| **Normalization** | Partition function $Z$ | Endogenous prior $P_i^0$ | Evidence lower bound | Attention denominator |
| **Constraint type** | Fixed mean energy $\langle E \rangle$ | Mutual information bound $I \leq \kappa$ | KL divergence from prior | Cross-entropy training loss |
| **Learning signal** | None (equilibrium) | Mutual information gradient | Precision-weighted error $\delta$ | Cross-entropy gradient |
| **Prior** | Maximum entropy (flat) | Endogenous $P_i^0$ | Generative model $P(s,y)$ | Positional encoding |
| **Domain** | Physical microstates | Economic actions | Sensory channels / motor actions | Sequence positions |
| **Founding paper** | Boltzmann 1877 | Matějka-McKay 2015 (AER) | Friston 2010 (Nat Rev Neurosci) | Vaswani et al. 2017 (NeurIPS) |

The four derivations differ in physical interpretation and derivation path. They are identical in mathematical structure. All four solve the same problem: allocate a bounded resource across a distribution of alternatives in a way that maximizes a value function subject to an entropy constraint. The unique solution is always the softmax at the appropriate temperature.

---

## V. The Entropy Equivalence

The deepest unification is the entropy equivalence: all four derivations are instances of the abstract problem

$$\max_\pi \; \mathbb{E}_\pi[u] + \tau \cdot H(\pi)$$

where $H(\pi) = -\sum_i \pi_i \log \pi_i$ is the Shannon entropy of the allocation distribution and $\tau \geq 0$ is the entropy weight. The unique solution to this problem is $\pi^* = \text{softmax}(u/\tau)$.

The mapping:
- **Statistical mechanics**: maximize entropy subject to $\langle E \rangle = \text{const}$ → dual of the Lagrangian is $\max_\pi H(\pi) - (1/kT) \langle E \rangle$ → same structure with $\tau = kT$, $u_i = -E_i$
- **Rational inattention**: maximize utility minus information cost → $\max_\pi \mathbb{E}_\pi[v] - \lambda \cdot I(A;O)$ → same structure with $\tau = \lambda$
- **Free energy principle**: minimize $\mathcal{F}$ → equivalent to $\max_\pi \mathbb{E}_\pi[-\mathcal{F}] + \tau H(\pi)$ under entropy regularization with $\tau = 1/\Sigma$
- **Entropy-regularized RL** (Ziebart et al., 2008; Haarnoja et al., 2018): explicitly adds $\alpha H(\pi)$ to the RL objective and derives $\pi^* = \text{softmax}(Q/\alpha)$ as the unique optimum
- **Cross-entropy training**: the Transformer's training objective minimizes $-\log P_\theta(y)$ subject to the implicit entropy of the softmax output distribution

Five derivations now, across five distinct research programs, all converging on the same object. The entropy-regularized objective is not a modeling choice. It is the canonical form of bounded-attention optimization, and the softmax is its solution.

Hyvärinen's (2005) score matching result closes a further loop: the score function $\nabla \log p_t(x)$ in diffusion models — the object that denoising score matching estimates — is the gradient of the log-likelihood, which is the gradient of the negative Fristonian free energy. Denoising in diffusion models is, mathematically, one step of free energy gradient descent, which is one step of the RI belief update. AlphaFold 3's diffusion module performs 200 such steps per structure — 200 consecutive precision-weighted prediction error updates, each implementing the Schultz (1997) dopaminergic update rule in silicon.

---

## VI. The Geometric Extension: Euclidean Miscalibration and the Hyperbolic Correction

All four derivations assume Euclidean geometry in the specification of the value function. In the Transformer, the value function is the Euclidean dot product $q_m \cdot k_t$. In RI, utility is typically linear in the state. In the free energy principle, prediction errors are measured in Euclidean space. The Boltzmann derivation makes no geometric assumption, but its standard physical implementation uses Euclidean inner products.

Robinson et al. (2024) measured Ricci curvature distributions across token neighborhoods in trained decoder-only language models and found substantial, variable negative curvature. Token frequency obeys a power law — a signature of hyperbolic geometry: high-frequency tokens cluster near the center of the Poincaré disk $\mathbb{B}^n$; rare, specific tokens lie near the boundary. The volume of a ball in hyperbolic space grows exponentially with radius, matching the power-law distribution of token frequency. Euclidean architecture must distort this geometry to represent it. HELM's experiments (NeurIPS 2025) confirm the cost: not respecting the geometry induces training instabilities and degrades generative performance. Correcting it — replacing the Euclidean dot product with the Minkowski inner product $\langle x, y \rangle_L = x_0 y_0 - \sum_i x_i y_i$ — recovers up to 4% gains on MMLU and ARC benchmarks at billion-parameter scale.

The economic and cognitive interpretation, which HELM does not provide, is the following. The Euclidean bounded-attention operator is the RI-optimal policy for a flat information space. The hyperbolic bounded-attention operator is the RI-optimal policy for a curved information space:

$$P^*_{\text{hyp}}(t|m) = \frac{\exp(\langle q_m, k_t \rangle_L / \sqrt{d_k})}{\sum_l \exp(\langle q_m, k_l \rangle_L / \sqrt{d_k})}$$

Any agent — biological or artificial — processing information whose natural geometry is hyperbolic (power-law distributed, hierarchically structured, rare-concept-rich) and using a Euclidean attention operator is systematically miscalibrated. The miscalibration is not random; it is directional. Euclidean RI underweights signals near the Poincaré disk boundary (rare, private, specific) and overweights signals near the center (common, public, generic) relative to what hyperbolic RI prescribes. This directional bias is the geometric origin of the finding examined in Novel Finding II below.

HELM-MiCE's Mixture-of-Curvature extension routes tokens to experts in distinct curvature spaces. In RI terms, this is the optimal policy for an agent whose information environment is a mixture of flat-geometry and curved-geometry signals — a heterogeneous attention market. CORDIC hardware (CARMEN, arXiv:2605.06878, May 2026) evaluates both Euclidean and Minkowski inner products natively in the same shift-and-add datapath, making it the natural hardware substrate for mixed-geometry bounded-attention computation. The hardware and the theory share the same unifying primitive: the rotation, which is circular in Euclidean space and hyperbolic in Minkowski space.

---

## VII. Novel Findings

The four-derivation convergence and the geometric extension establish the framework. The following findings are implications that emerge from this synthesis that have not been previously stated in the form given here.

---

### Finding I: The Information Temperature as a Cross-Domain Measurable Quantity

The temperature $\tau$ — $kT$ in thermodynamics, $\lambda$ in RI, $1/\Sigma$ in predictive coding, $\sqrt{d_k}$ in the Transformer — is the single free parameter of the bounded-attention operator. It governs the precision with which the system tracks value differences: low $\tau$ means sharp allocation (high precision, selective attention); high $\tau$ means diffuse allocation (low precision, distributed attention).

In physical systems, $\tau$ is set by environmental thermal contact. In engineered systems, it is set by architectural choice. In biological systems, it is set by developmental history — the cumulative effect of learning experiences, cultural training, and the objective functions that have governed the system's learning trajectory. The information temperature of a cognitive system is not fixed at birth. It is adjustable through the learning environments to which the system is exposed.

The novel finding is that the temperature parameter across these domains is not merely formally equivalent — it is empirically measurable in each domain and the measurements should satisfy cross-domain consistency constraints. A biological agent whose neural precision $\Sigma$ has been measured via fMRI or EEG in a given perceptual task should exhibit an RI cost $\lambda$ in economic choice tasks from the same domain that is consistent with their neural precision, and a behavioral response entropy in that domain that is consistent with both. An agent trained with Transformer temperature $\tau$ on a given corpus should exhibit, in human-interpretability analyses, response sharpness patterns consistent with that temperature.

Violations of cross-domain temperature consistency would identify agents whose attention mechanisms differ structurally across domains — agents who deploy high-precision attention in one modality and low-precision in another. This diagnostic has not been previously formulated. The information temperature is measurable, consequential, and adjustable. It is the central free parameter of human cognitive capacity, and its systematic manipulation through deliberate exposure to high-prediction-error learning environments is the mechanism through which developmental trajectories diverge.

---

### Finding II: Hyperbolic Rational Inattention and the Geometric Origin of the Morris-Shin Overweighting Bias

Morris and Shin (2002) demonstrated that rational agents overweight public signals relative to their information content, because each agent — uncertain about what others know — benefits from acting on information that others can also be expected to act on. The coordination value of a public signal adds to its informational value, producing systematic overweighting that persists in equilibrium. This is a foundational result of information economics, and it has been accepted as a consequence of strategic complementarities.

The geometric synthesis reveals a deeper explanation that operates independently of strategic complementarities and is testable at the individual level. Public signals in economic and social environments are, by definition, widely shared — they are common knowledge, low-surprise, high-frequency. In the geometry of the information space, they correspond to tokens near the center of the Poincaré disk: ubiquitous, easy to predict, low curvature. Private signals are rare, specific, low-frequency — near the boundary of the Poincaré disk.

An agent using Euclidean RI to process a hyperbolic information environment systematically underweights boundary signals (rare, private, high-surprise) relative to what the hyperbolic RI-optimal policy prescribes, and overweights center signals (common, public, low-surprise). This is the Morris-Shin overweighting result, derived from first principles of geometric miscalibration rather than from strategic behavior. The implication: Morris-Shin overweighting should occur even in the absence of any strategic environment, among agents who are processing the information privately and non-strategically, if those agents are applying Euclidean RI to a hyperbolic information space.

The prediction is specific and testable. Agents who demonstrate the Morris-Shin overweighting pattern in experimental settings should also demonstrate Euclidean bias in their internal information representations, measurable via the Caplin-Dean (2015) revealed preference characterization of RI behavior. Correcting the geometry — training agents to process hierarchical information in a hyperbolic representational space — should attenuate Morris-Shin overweighting independently of the strategic environment. Strategic complementarities and geometric miscalibration are not competing explanations. They are the same phenomenon at different levels of description, and the geometric level makes the structural prediction that the strategic level cannot: the overweighting is a calibration error, not merely a strategic equilibrium, and it is correctable.

---

### Finding III: Sleep as the Nightly Retroactive RI Reallocation Mechanism

The bounded-attention operator operates under a real-time channel capacity constraint during waking cognition. Signals arrive continuously; capacity is allocated in real time; signals with unexpectedly high information value generate large prediction errors but may have received insufficient channel capacity during the waking period — not because they are unimportant, but because real-time processing constraints prevent retroactive reallocation.

Wilson and McNaughton (1994) showed that hippocampal neurons active during novel environment exploration replay their activation sequences during subsequent sleep, at 10–20× compressed speed. Stickgold and Walker (2013) showed that sleep selectively replays high-novelty, high-prediction-error events over familiar, confirmatory ones. Born and colleagues (2019) found that hippocampal replay during slow-wave sleep occurs in reverse sequence order — the end of an experience replays before its beginning.

The synthesis: sleep is the biological implementation of retroactive RI reallocation. During the waking period, the brain allocates channel capacity under real-time constraints and generates prediction errors that flag which signals were under-processed given their information value. Sleep removes the real-time constraint. The hippocampus replays prediction-error-weighted events — precisely the events for which the gap between information value and capacity allocated was largest — to the cortex, which can devote full processing resources to integrating them into existing schema structures. The reverse temporal order of replay implements the RI credit-assignment operation: it allows the brain to update not only the representation of the surprising event, but the predictions that generated the mis-prediction.

Dewar et al. (2012) demonstrated that fifteen minutes of quiet wakeful rest immediately following learning improves recall by 10–15%, even without sleep. This identifies the first stage of a three-stage RI reallocation process: (1) prediction error during waking encoding flags high-value under-processed signals; (2) immediate wakeful rest allows shallow reallocation — brief capacity extension to the flagged signals without full schema integration; (3) subsequent slow-wave sleep enables deep reallocation through hippocampal reverse replay — full schema integration and credit assignment to preceding predictions.

The educational implication is that what follows learning is as computationally consequential as the learning event itself. Massed practice that eliminates inter-session gaps eliminates the reallocation windows in which the waking prediction errors generated during study are processed into durable schema updates. The desirable difficulty research (Bjork and Bjork, 2011) is not a separate finding from the sleep research. They are the front end and back end of the same RI processing cycle: desirable difficulties maximize the prediction errors generated during encoding; sleep maximizes the reallocation depth applied to those prediction errors. Both are required; neither is sufficient alone.

---

### Finding IV: Desirable Difficulties as Forced High-Value RI Sampling

Bjork and Bjork (2011) documented six conditions — spacing, interleaving, retrieval practice, varied practice, generation, reduced feedback — that impede immediate performance while enhancing long-term retention and transfer. The neuroscientific account identifies prediction error as the mechanism. The RI account is more precise and more generative.

Each desirable difficulty is a precision-targeted mechanism for maximizing mutual information extracted per unit of channel capacity by ensuring that the signals processed carry near-maximum prediction error. Spacing prevents confirmatory re-exposure from consuming capacity on near-zero-information signals — the prior has partially decayed, so the incoming signal produces a larger gap between prior and posterior, extracting more information per encoding event. Interleaving forces maximum prediction error at every step: the prior built by the preceding item is misaligned with the current item's category, generating a full prediction error rather than a confirmatory signal. Retrieval practice forces the agent to generate a prediction before receiving the signal, guaranteeing that the signal produces a non-trivial prediction error regardless of its content. Generation forces commitment: the agent who has committed to a prediction cannot process the correct answer as confirmatory, regardless of whether the prediction was correct.

The fifteen-to-twenty percentage point improvement in Kornell et al.'s (2009) wrong-answer experiment — where generating an incorrect answer before seeing the correct one produces 15–20% better final recall than studying the answer directly — is the behavioral signature of approximately doubling the mutual information extracted per encoding event. The wrong-answer generation forces a full prediction error on every item; direct study produces a near-zero confirmatory signal on every item. The difference in retained information is the product of the difference in prediction error magnitude across all items.

The compounding property follows directly from RI theory and has not been previously stated in this form. Each high-prediction-error encoding event updates the prior, building a richer, more differentiated schema. A richer prior generates *larger* prediction errors for subsequent inputs at the same nominal difficulty, not smaller — because the richer schema has sharper expectations that are more precisely violated by incoming information. The learner trained under desirable difficulties becomes progressively more capable of extracting information from difficult material, because their prior is better calibrated to generate productive prediction errors per unit of channel capacity. The compounding is structural, not incidental: high-prediction-error encoding permanently lowers the information temperature of the system for the relevant domain, enabling higher-precision bounded attention allocation going forward.

---

### Finding V: The Sherman-Morrison Identity as the Rank-One RI Update

The Sherman-Morrison identity (1950) provides the closed-form update to a matrix inverse when the matrix changes by a rank-one perturbation:

$$(\mathbf{F} + \mathbf{u}\mathbf{v}^T)^{-1} = \mathbf{F}^{-1} - \frac{\mathbf{F}^{-1}\mathbf{u}\mathbf{v}^T\mathbf{F}^{-1}}{1 + \mathbf{v}^T \mathbf{F}^{-1} \mathbf{u}}$$

This is the unique efficient solution to the following problem: given an existing attention allocation encoded in $\mathbf{F}^{-1}$, compute the optimal updated allocation given that exactly one coordinate of the information environment has changed by $\mathbf{u}\mathbf{v}^T$. The computational cost is one outer product plus one normalization — two operations — rather than the full matrix inversion that would be required to recompute the optimal allocation from scratch.

The connection to RI theory is exact, not approximate. In the RI framework, the attention allocation at any moment is a function of the current information environment. When the environment changes by exactly one coordinate — one new signal, one new observation, one new state variable — the optimal Bayesian update to the RI attention allocation is the Sherman-Morrison rank-one update applied to the precision matrix. The general case (arbitrary information update) requires full re-inference; the rank-one case has an exact, computationally cheap closed form.

The implication for computational biology extends to the RI framework the insight that the Matmul Ceiling document (ERI Labs, 2026) identifies architecturally. A CRISPR saturation mutational scan evaluates variants differing from wild-type by one base substitution — each variant is a rank-one perturbation of the genomic information tensor. The Sherman-Morrison RI update is the computationally optimal method for computing the attention reallocation for each variant. On a rank-one-update-native substrate, the unit cost per variant is one outer product and one normalization rather than a full re-inference from the updated sequence: an 80–100× reduction in compute that is not a hardware optimization but a mathematical identity. The biological problem and the economic problem share the same mathematical core.

More broadly: the Sherman-Morrison identity is the RI update rule for agents operating in environments that change continuously but locally — one variable at a time. This is the structure of most real information environments. Drug response changes when one gene variant changes. Market conditions change when one policy instrument changes. A conversation changes when one new sentence is added. The rank-one RI update is not a special case; it is the standard case of environmental change, and its exact closed-form solution has been available since 1950 without its RI interpretation.

---

### Finding VI: Abduction as the Mechanism of RI Prior Extension

Peirce (1903) distinguished deduction (what must follow from premises), induction (what the evidence shows to be operative), and abduction (what hypothesis would explain the evidence if true). Only abduction introduces genuinely new concepts — new hypotheses not previously in the consideration set.

In RI terms, deduction and induction are operations that allocate channel capacity within a fixed consideration set $\mathcal{A}$: they compute $P^*(a|s)$ for $a \in \mathcal{A}$ where $\mathcal{A}$ is given. Abduction is the meta-level RI operation of extending $\mathcal{A}$ — allocating initial channel capacity to a hypothesis not previously represented in the prior. The channel capacity cost of abduction is the cost of processing a signal class whose prior probability was zero: it requires a structural update to the RI problem itself, not merely a belief update within the existing problem. This cost is always high, because zero-prior hypotheses carry no prior capacity allocation, and the first allocation to them must come at the expense of previously allocated capacity elsewhere.

The novel finding is that abduction is optimal RI expansion: when the existing consideration set fails to explain incoming evidence with sufficient precision — when the free energy under the best available hypothesis exceeds a threshold — the RI-optimal response is to expand $\mathcal{A}$ rather than allocate more capacity within it. Abduction is not a separate logical faculty; it is the RI operation that fires when within-set precision is saturated.

The agents for whom abductive RI expansion is most accessible are those whose existing consideration set is large enough and whose prior structural library is rich enough that the cost of adding a new hypothesis is low — because the new hypothesis can be anchored to existing schema structures rather than built from zero. This is the formal mechanism for the cross-domain inference documented in THE-DEDUCTION-WINDOW (ERI Labs, 2026): completing the eight-step inference from the Transformer paper's hardware section to programming credential obsolescence required allocating channel capacity to the hypothesis that a natural language processing architecture had implications for labor economics. This was an abductive RI expansion across a structural hole between two domains. Agents whose prior structural library spanned both domains could complete the expansion at low channel capacity cost. Agents whose priors were domain-specific could not allocate capacity across the structural hole without exceeding their channel capacity budget.

Kapoor et al. (2026) identify frame construction — the generation of a problem representation from an underspecified situation — as the primary failure mode of frontier AI systems. Frame construction is the AI instantiation of abductive RI expansion: generating the consideration set before optimization begins. The bounded-attention operator is optimally efficient within a fixed frame; it cannot generate the frame. The agents trained to extend their consideration sets through cross-domain structural exposure hold the cognitive operation that the operator's own architecture cannot provide for itself.

---

### Finding VII: The Gabaix Temporal Discount as RI Applied Across Time

Gabaix (2020) derives the behavioral discount factor $M^k$ — the degree of attention agents pay to events $k$ periods in the future — from RI theory: attending to distant future states consumes channel capacity that competes with near-future states, and the RI-optimal agent discounts future attention exponentially because extending the temporal horizon accumulates capacity costs multiplicatively. The formal result is that future attention decays as $M^k$ where $0 < M \leq 1$, producing what Gabaix calls the near-sighted Phillips curve and the discounted Euler equation.

The novel finding is that $M$ is the RI temperature applied across the temporal dimension. The standard RI temperature $\tau$ governs the precision of attention allocation across states at a given moment; $M$ governs the precision of attention allocation across time horizons. High $M$ (close to 1, shallow temporal discounting) corresponds to a low temporal RI temperature — low cost of processing long-horizon signals, high precision of long-horizon attention. Low $M$ (steep discounting) corresponds to a high temporal RI temperature — high cost of processing long-horizon signals, diffuse long-horizon attention.

The compound-interest implication: an agent with low temporal RI temperature ($M$ near 1) accumulates compounding returns from long-horizon attention allocation. Each long-horizon inference enables earlier developmental investment that compounds with time, enabling further long-horizon inferences at lower cost because the prior is better calibrated to generate productive prediction errors about future states. An agent with high temporal RI temperature is limited to near-horizon processing, which cannot generate the compounding structure available from long-horizon signals.

The eight-step inference from Vaswani et al. (2017) to programming credential obsolescence required maintaining a probabilistic belief — that the architecture implied a specific labor market consequence within one professional generation — for three years before formal confirmation (Kaplan et al., 2020) and six years before labor market confirmation. This maintenance required $M$ close to 1: a temporal RI temperature low enough that the long-horizon signal remained in the consideration set at non-trivial probability weight throughout the confirmation window, despite accumulating near-horizon evidence about the current value of programming credentials. The credential-optimized agent's high temporal RI temperature meant the signal was discounted out of their consideration set before confirmation arrived. The economics outlier's low temporal RI temperature meant the signal remained alive and available for updating.

The educational implication is that temporal RI temperature — the $M$ parameter of the cognitive system — is set by the objective functions used to train the system's attention allocation. A curriculum organized around near-horizon credential acquisition trains a high temporal RI temperature: the rewarded horizon is always at most one semester away. A curriculum organized around intertemporal optimization under uncertainty trains a low temporal RI temperature: the rewarded horizon is explicitly long, and the student is evaluated on their ability to maintain credible beliefs about uncertain future states across extended confirmation windows. The economics curriculum trains the low-$M$ practitioner by design. The credential curriculum trains the high-$M$ practitioner by default.

---

## VIII. The Unified Theorem and Its Implications

The seven findings converge on a statement.

The bounded-attention operator — the softmax at temperature $\tau$ — is the unique solution to entropy-regularized expected-value maximization, and this problem characterizes optimal allocation across any domain in which a bounded resource must be distributed over alternatives. Statistical mechanics, economics, neuroscience, and computer science have each discovered this independently. Their convergence is proof that the result is domain-neutral: it is a theorem about the structure of optimization under bounded capacity, not a fact about any particular physical, cognitive, or computational system.

The temperature $\tau$ is the central free parameter of this theorem in biological systems. Unlike physical systems (where temperature is environmental) and engineered systems (where temperature is architectural), biological cognitive systems acquire their information temperature through developmental history. Systems exposed to high-prediction-error learning environments across diverse domains acquire low information temperature: they allocate channel capacity sharply to high-value signals, maintain long-horizon beliefs at non-trivial probability weight, and generate large prediction errors — and therefore large learning signals — on incoming information. Systems exposed to low-prediction-error confirmatory environments acquire high information temperature: diffuse attention, steep temporal discounting, small prediction errors, small learning signals.

Desirable difficulties reduce information temperature by forcing high-prediction-error encoding events. Sleep deepens this reduction by enabling retroactive reallocation of capacity to the signals whose prediction errors were largest. Abductive cross-domain exploration reduces the channel capacity cost of extending the consideration set, enabling RI expansion at lower cost. Long-horizon intertemporal training reduces temporal RI temperature. Together, these form the biological program for building a bounded-attention system operating at or near the theoretical optimum of the operator it instantiates.

The hyperbolic correction — replacing Euclidean with Minkowski inner products in the value function — is the geometric calibration required when the information environment is not flat. Natural language, social information, genomic variation, and hierarchically structured knowledge all exhibit power-law distributions and negative curvature that the Euclidean operator misrepresents. The 4% performance gain of HELM on general language benchmarks is the empirical cost of geometric miscalibration at billion-parameter scale. In biological cognitive systems, the corresponding cost is the Morris-Shin overweighting bias — systematic underattention to rare, boundary, private signals that is a predictable consequence of Euclidean RI applied to a hyperbolic information environment.

The system that has acquired low information temperature through deliberate high-prediction-error exposure, been calibrated to the hyperbolic geometry of its information environment, and extended its consideration set through cross-domain abductive practice is the cognitive architecture that most closely approximates the theoretical optimum of the bounded-attention operator. It is not a more intelligent system in any domain-specific sense. It is a more precisely calibrated bounded-attention operator — one whose temperature, geometry, and consideration-set structure more closely match the structure of the information environments it processes.

This architecture is trained by exactly the developmental conditions that most credential-optimizing educational systems select against: productive failure under desirable difficulties, cross-domain structural exposure, abductive hypothesis extension, long-horizon intertemporal orientation under uncertainty, and the tolerance to maintain probabilistic inferences across extended confirmation windows. The conditions are not pedagogically convenient. They are the precise engineering requirements for building a bounded-attention system operating at its theoretical optimum — and the bounded-attention operator is the mathematical object that all of neuroscience, economics, physics, and computer science have been independently converging on for 150 years.

The softmax was in Boltzmann in 1877.  
The softmax was in the American Economic Review in 2015.  
The softmax was in NeurIPS in 2017.  
The softmax is in every dopaminergic synapse that has ever fired.

The convergence was not discovered. It was always there, waiting for a frame large enough to contain it.

---

## Research Foundation

| Source | Core Finding Applied |
|:---|:---|
| Boltzmann, L. (1877). Über die Beziehung zwischen dem zweiten Hauptsatze der mechanischen Wärmetheorie und der Wahrscheinlichkeitsrechnung. *Sitzungsberichte der Akademie der Wissenschaften*, 76, 373–435. | The first derivation of the softmax — the bounded-attention operator — from entropy maximization subject to a mean-energy constraint; the thermodynamic instance that predates the economic and computational derivations by 126–140 years; identified here as the founding proof of the theorem that the RI literature, the Transformer literature, and the free energy literature each independently rediscover |
| Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423. | Channel capacity and mutual information as the information-theoretic vocabulary shared by the RI and machine learning derivations; the common ancestor of the economic and computational branches of the convergence, published 71 years after Boltzmann's thermodynamic proof of the same fundamental result |
| Sims, C.A. (2003). Implications of rational inattention. *Journal of Monetary Economics*, 50(3), 665–690. | Shannon capacity imported into macroeconomics; optimal behavior under bounded cognition as mutual-information-constrained utility maximization; the foundational RI paper establishing that the bounded-attention operator is the correct model of economic information processing |
| Matějka, F., & McKay, A. (2015). Rational inattention to discrete choices: A new foundation for the multinomial logit model. *American Economic Review*, 105(1), 272–298. | The proof that the RI-optimal discrete choice policy is the softmax — formally identical to the Transformer's attention mechanism — derived from first principles of constrained utility maximization two years before the Transformer's publication; the central exhibit in the four-derivation convergence |
| Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30. | The machine learning derivation of the bounded-attention operator; the empirical discovery that the RI-optimal policy, when implemented as scaled dot-product attention, generalizes to every sequence domain; the architectural implementation of the Matějka-McKay (2015) result without knowledge of that result |
| Friston, K.J. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138. | Free energy minimization as the unifying principle of neural computation; precision as the attention operator over sensory channels; the neuroscientific derivation of the bounded-attention operator from Bayesian inference under biological constraints; the third independent proof of the universal optimality theorem |
| Clark, A. (2016). *Surfing Uncertainty: Prediction, Action, and the Embodied Mind.* Oxford University Press. | Extension of the free energy principle across perception, action, and social cognition; the brain as a hierarchical prediction machine whose learning rule is the RI-optimal policy expressed as precision-weighted prediction error; the theoretical framework connecting Friston's math to the behavioral and educational literatures |
| Schultz, W., Dayan, P., & Montague, P.R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593–1599. | Dopaminergic prediction-error coding: the biological learning signal is proportional to the gap between predicted and received reward, not to reward magnitude; empirical confirmation that the brain's update rule is the RI gradient — the gradient of the mutual information objective — and not a direct reward signal |
| Jaynes, E.T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4), 620–630. | The formal identification of Boltzmann-Gibbs statistics as maximum entropy inference under energy constraints; the bridge between statistical mechanics and information theory that makes the Boltzmann-to-Shannon-to-RI derivation chain explicit; the proof that the thermodynamic derivation and the information-theoretic derivation are the same argument |
| Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML*. | Explicit entropy regularization in reinforcement learning: adding $\alpha H(\pi)$ to the RL objective produces $\pi^* = \text{softmax}(Q/\alpha)$; the fifth independent derivation of the bounded-attention operator, from empirical machine learning rather than first principles, confirming the entropy equivalence across fields |
| Ziebart, B.D., Maas, A., Bagnell, J.A., & Dey, A.K. (2008). Maximum entropy inverse reinforcement learning. *AAAI*. | Maximum entropy formalization in inverse RL; the early machine learning confirmation of the entropy equivalence connecting rational inattention, Boltzmann statistics, and optimal sequential decision-making under uncertainty |
| Robinson, J., et al. (2024). The structure of the token space for large language models. arXiv:2410.08993. | Empirical measurement of negative Ricci curvature in LLM token embeddings and power-law token frequency distribution; the geometric evidence establishing that the information space of natural language is hyperbolic, not Euclidean — the empirical motivation for Finding II and the hyperbolic RI framework |
| HELM / HELM-MiCE. Hyperbolic Large Language Models via Mixture-of-Curvature Experts. *NeurIPS 2025*, arXiv:2505.24722. | Billion-parameter hyperbolic LLM achieving up to 4% gains on MMLU and ARC over Euclidean architectures; the empirical confirmation that the hyperbolic bounded-attention operator outperforms the Euclidean operator on natural language tasks; Finding II applied at production scale |
| Morris, S., & Shin, H.S. (2002). Social value of public information. *American Economic Review*, 92(5), 1521–1534. | Public signal overweighting due to coordination premium; Finding II provides the geometric reinterpretation: Euclidean RI's systematic overweighting of common/central signals relative to rare/boundary signals is the Morris-Shin result derived from geometric miscalibration rather than strategic complementarities alone; testable prediction: the overweighting should persist in non-strategic settings and attenuate with hyperbolic RI calibration |
| Wilson, M.A., & McNaughton, B.L. (1994). Reactivation of hippocampal ensemble memories during sleep. *Science*, 265(5172), 676–679. | Hippocampal replay: neurons active during novel exploration replay their sequences during subsequent sleep with preferential replay of novel over familiar environments; the behavioral evidence for Finding III: sleep as the biological mechanism of retroactive RI reallocation |
| Stickgold, R., & Walker, M.P. (2013). Sleep-dependent memory triage: evolving generalization through selective processing. *Nature Neuroscience*, 16(2), 139–145. | Sleep-dependent selective consolidation: prediction-error-weighted replay during slow-wave sleep; the formal connection to RI: the replay priority function is the RI under-processing signal — large prediction error indicates large gap between information value and capacity allocated during waking encoding |
| Born, J., et al. [Tübingen group, 2019]. Reverse replay during sharp-wave ripples in slow-wave sleep. | Reverse temporal order of hippocampal replay during slow-wave sleep; consistent with Finding III: the retroactive RI reallocation mechanism assigns credit to the predictions that preceded surprising outcomes, not only to the outcomes themselves — the backward replay implements the RI credit-assignment operation |
| Dewar, M., Alber, J., Butler, C., Cowan, N., & Della Sala, S. (2012). Brief wakeful resting boosts new memories over the long term. *Psychological Science*, 23(9), 955–960. | Wakeful rest effect: 10-15% recall improvement from 15 minutes of quiet rest immediately post-encoding; the first stage of the three-stage RI reallocation process identified in Finding III, beginning before sleep is possible |
| Nader, K., Schafe, G.E., & LeDoux, J.E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406(6797), 722–726. | Memory reconsolidation: retrieved memories are labile for ~6 hours before re-storage in updated form; the RI interpretation: retrieval opens a high-bandwidth reallocation window in which the current prior — updated by all learning since original encoding — is applied to re-encode the original signal with updated precision |
| Bjork, R.A., & Bjork, E.L. (2011). Making things hard on yourself, but in a good way. In *Psychology and the Real World.* Worth Publishers. | Desirable difficulties: six conditions (spacing, interleaving, retrieval practice, varied practice, generation, reduced feedback) that maximize prediction error per encoding event; Finding IV identifies these as precision-targeted mechanisms for maximizing mutual information extracted per unit of channel capacity |
| Kornell, N., Hays, M.J., & Bjork, R.A. (2009). Unsuccessful retrieval attempts enhance subsequent learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 35(4), 989–998. | Generation effect: 15–20% improvement in final recall for wrong-answer-then-correct vs. direct study; the behavioral signature of approximately doubling mutual information extracted per encoding event by forcing full prediction error at every item |
| Kornell, N., & Bjork, R.A. (2008). Learning concepts and categories: Is spacing the "enemy of induction"? *Psychological Science*, 19(6), 585–592. | Interleaving superiority: 43-percentage-point gain on novel-stimulus far transfer; the RI interpretation: interleaving forces maximal prediction error at every step by ensuring prior-item schema misalignment at every encoding event |
| Sherman, J., & Morrison, W.J. (1950). Adjustment of an inverse matrix corresponding to a change in one element of a given matrix. *Annals of Mathematical Statistics*, 21(1), 124–127. | Rank-one matrix inverse update: the closed-form solution to the RI precision update when exactly one coordinate of the information environment changes; Finding V: the Sherman-Morrison identity is the rank-one RI update, with the CRISPR variant scanning problem as its direct biological application |
| Peirce, C.S. (1903). Lectures on Pragmatism. *Collected Papers*, Vol. V. | Abduction as the only logical operation that introduces genuinely new ideas; Finding VI: abduction is the RI operation of extending the consideration set — the meta-level operation whose channel capacity cost determines which agents can complete cross-domain inferences and which cannot; the cognitive operation at the boundary of AI capability identified by Kapoor et al. (2026) |
| Gabaix, X. (2020). A behavioral New Keynesian model. *American Economic Review*, 110(8), 2271–2327. | Behavioral discount factor $M^k$: exponential decay of attention to future states derived from RI theory; Finding VII: $M$ is the temporal RI temperature — the parameter governing the precision with which a cognitive system allocates channel capacity to long-horizon signals; the compounding advantage of low temporal RI temperature |
| Caplin, A., & Dean, M. (2015). Revealed preference, rational inattention, and costly information acquisition. *American Economic Review*, 105(4), 1468–1493. | Revealed preference characterization of RI behavior from choices; the theoretical framework for Finding II's testable prediction that Morris-Shin overweighting should correlate with Euclidean bias in internal representations, measurable from choice behavior alone |
| Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *Journal of Machine Learning Research*, 6, 695–708. | Score function $\nabla \log p_t(x)$ as the gradient of the log-likelihood; the score function in diffusion models = gradient of the Fristonian negative free energy; denoising score matching = one step of RI belief update = one step of the Schultz (1997) dopaminergic prediction error update; unifying diffusion model denoising with the biological learning rule |
| Kapoor, S., Kirgis, P., Schwartz, A., et al. (2026). Open-world evaluations for measuring frontier AI capabilities. arXiv:2605.20520. | Frame construction as the primary AI failure mode; the capability boundary falls at the abductive RI expansion operation — extending the consideration set before optimization begins; empirical confirmation that Finding VI predicts the exact location of the human-AI cognitive frontier |
| Peterson, A.J. (2025). Training for obsolescence? University of Poitiers. arXiv:2508.19625. | Teachability-substitutability correlation: skills fully specifiable as curriculum objectives are maximally learnable by the bounded-attention operator operating over a fixed training distribution; the RI interpretation: skills whose optimal execution is a softmax over a fixed consideration set are parameterizable training objectives and therefore substitutable |
| Molavi, P. (2019). Macroeconomics with learning and misspecification. *Quarterly Journal of Economics*, 134(3), 1421–1484. | Model-uncertainty-aware learners outperform Bayesian optimal learners at structural breaks; the economics curriculum's model-uncertainty default as a mechanism for maintaining low temporal RI temperature ($M$ near 1) across regime changes — a direct consequence of Finding VII |
| Dunlosky, J., et al. (2013). Improving students' learning with effective learning techniques. *Psychological Science in the Public Interest*, 14(1), 4–58. | Large-scale efficacy evaluation: practice testing and distributed practice rated highest across all ages and domains; re-reading rated lowest; the systematic empirical confirmation that RI-theoretically predicted high-prediction-error techniques dominate confirmatory techniques in all measured comparisons |
| Ericsson, K.A., Krampe, R.T., & Tesch-Römer, C. (1993). The role of deliberate practice in the acquisition of expert performance. *Psychological Review*, 100(3), 363–406. | Expert performance produced by hours at the current performance edge, not by total hours; in RI terms: deliberate practice is the sustained maintenance of high prediction error per unit of channel capacity — the information-temperature reduction program applied systematically over career-length timescales |
| Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361. | Power-law scaling of language model performance with compute, data, and parameters; the formal confirmation of the scaling trajectory inferable from the Transformer's architecture in 2017; the three-year window between informal abductive inference and quantitative confirmation |
| van der Wijk, V., et al. (2026). Fast and geometrically grounded Lorentz neural networks. arXiv:2601.21529. | Distance-to-hyperplane formulation resolving the Lorentz linear layer norm-scaling instability; the closing of the central theoretical obstacle to deep hyperbolic training at scale; the production-readiness confirmation for hyperbolic RI architectures identified in Finding II |
| CARMEN. (2026). arXiv:2605.06878. | CORDIC hardware achieving 4.83 TOPS/mm² in 28nm CMOS with runtime-adaptive precision for both Euclidean and Minkowski operations; the hardware instantiation of the geometric extension in Section VI — a single datapath evaluating both the Euclidean and hyperbolic bounded-attention operators natively, without the split-brain VPU approximation of matmul-only substrates |
| Vaswani et al. 2017 (hardware section); Jouppi, N.P., et al. (2017). In-datacenter performance analysis of a Tensor Processing Unit. ISCA 2017. arXiv:1704.04760. | TPU v2 in May 2017, one month before Transformer submission: the hardware co-design signal indicating that the bounded-attention operator would be scaled to all sequence domains; the most information-valuable component of the founding paper under a long-horizon RI reading objective |

---

*ERI Labs · Eric Ren · Jersey City, New Jersey · github.com/ericrenone · 2026*

*Prior analyses in the ERIE-SCHOOL lineage: THE-COGNITIVE-TAX · POLITICAL-BYPASS · NARRATIVE-DISPOSSESSION · COMPETITIVE-EVACUATION · CAPACITY-SUPPRESSION · PROXIMITY-AS-WEAPON · THE-HIDDEN-SECTOR-OF-HUMAN-CAPACITY · DEVELOPMENTAL-HETEROGENEITY-AND-ITS-INSTITUTIONAL-ERASURE · CHRONOLOGICAL-TYRANNY · THE-EXIT-THEOREM · THE-COLLECTIVE-OUTLIER · THE-INNER-PARTITION · EMERGENT-INTELLIGENCE · THE-ANTERIOR-BYPASS · THE-PREEMPTIVE-OFFENSIVE · THE-KINSHIP-INVERSION · THE-NORMALIZATION-HORIZON · THE-WORKPLACE-TRANSFERENCE · THE-FIELD-CAPTURE · THE-COMPLETE-DEVELOPMENTAL-SUPPRESSION-TRAJECTORY · THE-SUPPRESSION-GENERATIVITY-ALIGNMENT · THE-EARLIEST-VERDICT · THE-FORECLOSED-MIND · THE-INTERIOR-RECORD · THE-CRYPTIC-PHENOTYPE · THE-PRE-ADAPTATION · THE-DEDUCTION-WINDOW · THE-ATTENTION-PREMIUM · THE-TRANSFORMATION-GRADIENT · THE-SOFTMAX-CONVERGENCE*
