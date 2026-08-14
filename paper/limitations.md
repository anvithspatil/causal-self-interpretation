
## 1. Small Evaluation Set

The final V5.2 experiment used only six held-out evaluation examples.
Therefore, the observed 100% alignment should be interpreted as a preliminary result rather than a statistically robust estimate of generalization performance.

A larger evaluation set is required to determine whether the observed alignment persists across substantially more examples.

---

## 2. Synthetic Task Environment

The experiments use deterministic artificial systems with binary variables A, B, and C.

This controlled environment makes the ground-truth causal variable known exactly, which is useful for mechanistic analysis.

However, synthetic tasks are substantially simpler than natural
language reasoning.

The results therefore cannot be assumed to generalize to complex reasoning, factual questions, planning, or real-world decision-making.

---

## 3. Single Model

The experiments were conducted using Qwen3-4B.

Consequently, the findings should not be interpreted as evidence about language models in general.

Testing additional models and model sizes would be necessary to determine whether the observed representation–self-report alignment is a broader phenomenon.

---

## 4. Representational Alignment Is Not Causal Grounding

The most important limitation is the distinction between information being represented and information being causally responsible for an output.

V5.2 demonstrates alignment between:

    verbal causal attribution

and

    activation-derived factor prediction.

It does not demonstrate that the identified representation is necessary or sufficient for the model's decision.

The activation-patching experiments in V5 and V5.1 did not provide sufficient evidence for such a causal claim.

Therefore, the strongest supported interpretation is representation–self-report alignment rather than causal self-interpretation.

---

## 5. Late-Layer Analysis

The V5.1 and V5.2 analyses focused on the final twelve transformer layers for computational efficiency.

Earlier layers were therefore not exhaustively evaluated in the final representation analysis.

This means the experiments cannot establish that the relevant information first emerges in the late layers.

---

## 6. Representation Construction

V5.2 represents each example using a mean activation across token positions.

This provides a compact representation but may discard information encoded at specific token positions or in relationships between tokens.

A more detailed analysis could investigate token-level, attention-level, or subspace-level representations.

---

## 7. Prototype-Based Classification

The internal causal factor in V5.2 is inferred by comparing an evaluation representation with factor-specific prototype
representations.

This is an operational definition of internal prediction.

It should not be interpreted as demonstrating that the model itself uses the same prototype structure during computation.

---

## 8. Self-Report Reliability

The verbal self-report is treated as an observable behavioral output of the model.

A correct self-report does not independently establish that the model possesses an explicit introspective mechanism.

The model may generate a correct causal explanation through learned patterns without possessing privileged access to the underlying computation.

---

## 9. Future Work

Several extensions could strengthen the study:

1. Increase the number of held-out evaluation examples.
2. Test multiple Qwen3 model sizes.
3. Test additional language models.
4. Evaluate natural-language reasoning tasks.
5. Perform more extensive layer and token localization.
6. Use causal interventions that target distributed representation
   subspaces rather than individual token positions.
7. Compare activation-derived predictions against stronger causal
   intervention baselines.
8. Test whether representation–self-report alignment remains stable
   under model modifications or fine-tuning.

---

## 10. Overall Limitation

The current study should therefore be viewed as a controlled proof-of-concept.

Its main contribution is not demonstrating that language models can perfectly introspect their own computations.

Instead, it demonstrates a reproducible experimental framework for
separating three questions:

    1. Can the model verbally identify a causal factor?
    2. Is information about that factor encoded in its activations?
    3. Is that representation causally responsible for the output?

The current experiments provide positive evidence for the first twoquestions in the controlled setting, while leaving the third unresolved.