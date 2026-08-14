# Results

## Experimental progression

| Experiment | Purpose | Evaluation n | Self-report | Causal grounding |
|---|---|---:|---:|---:|
| V5 | Activation intervention | 10 | 90.0% | 60.0% |
| V5.1 | Factor-specific cross-patching | 3 | 66.7% | 0.0% |
| V5.2 | Representation/self-report alignment | 6 | 100.0% | Not evaluated |

## V5

The recorded V5 evaluation contains 10 examples. Nine self-reports are correct and one is `UNKNOWN`, giving 90% self-report accuracy. Six of ten records satisfy the experiment's stored `causal_grounding` criterion.

The intervention result is mixed: some examples show substantial target restoration, while others show weak or negative restoration. This prevents treating V5 as definitive evidence of causal grounding.

## V5.1

V5.1 used 3 localization examples and 3 evaluation examples. The selected candidates were all in layer 34, at tokens 28, 24, and 29 for A, B, and C.

Recorded results:

- Self-report accuracy: 66.7%
- Causal-grounding rate: 0%
- Median specificity ratio: 0.5

Matched patch effects were 0.0 for example 4, 0.015625 for example 5, and 0.0 for example 6. The result does not provide convincing evidence that the localized representations were causally sufficient or factor-specific.

## V5.2

V5.2 changed the question from causal intervention to representational alignment. It used 6 localization examples and 6 held-out evaluation examples.

The most factor-separable layer was layer 34, with mean pairwise activation distance 17.8843.

| Metric | Result |
|---|---:|
| Self-report accuracy | 6/6 = 100% |
| Internal factor prediction | 6/6 = 100% |
| Self-report/internal alignment | 6/6 = 100% |

All six held-out examples had the same factor as the ground truth, the model's verbal self-report, and the activation-derived internal prediction.

## Interpretation

The strongest supported claim is:

> In this controlled synthetic setting, Qwen3-4B's verbal causal attributions aligned with factor-discriminative information recoverable from its internal representations on six held-out examples.

This does **not** establish causal introspection. In particular:

**Representational alignment ≠ causal grounding.**

The V5.1 intervention failed to establish causal sufficiency.

## Limitations

The final V5.2 evaluation contains only six held-out examples and uses one model. The tasks are synthetic and deterministic. The internal prediction is prototype-based and uses mean hidden activations. The result is therefore a proof-of-concept finding, not a general estimate of model introspection.

## Conclusion

The combined experiments support two cautious conclusions:

1. Internal activations contain information that distinguishes the causal factors in the controlled task.
2. That information aligns with the model's verbal causal attribution in the V5.2 held-out set.

Whether the identified representation is causally necessary or sufficient remains unresolved.
