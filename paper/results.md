
## 1. Overview

The experiments were carried out in multiple stages to distinguish between behavioral self-attribution, representational alignment, and causal intervention.

Three main observations emerged:

1. The model was able to identify the causal variable through verbal self-report.
2. The model's internal activation representations contained information that could distinguish between the causal factors.
3. The activation-patching experiments did not provide sufficient evidence that these identified representations were causally sufficient for the model's behavior.

---

## 2. V5 Baseline

The initial V5 experiment evaluated both verbal self-report and activation based intervention.

The evaluation consisted of 10 examples.

| Metric                        | Result |
| ----------------------------- | -----: |
| Evaluation examples           |     10 |
| Self-report accuracy          |    90% |
| Initial causal-grounding rate |    60% |

These intervention results were interpreted cautiously because the experimental design relied on a common late layer representation and did not establish factor specific causal sufficiency.

---

## 3. V5.1 Representation Specificity

V5.1 investigated whether the representations associated with factors A, B, and C were causally specific.

The experiment used separate representations for each factor and performed cross factor activation patching.

The initial smoke evaluation consisted of three evaluation examples.

| Metric                   | Result |
| ------------------------ | -----: |
| Localization examples    |      3 |
| Evaluation examples      |      3 |
| Self-report accuracy     |  66.7% |
| Causal-grounding rate    |     0% |
| Median specificity ratio |    0.5 |

The matched activation patches did not consistently produce stronger behavioral effects than the mismatched patches.

Therefore, the results do not support interpreting the localized activation differences as causally sufficient representations.

---

## 4. V5.2 Representation–Self-Report Alignment

V5.2 shifted the research question from causal sufficiency to representational alignment.

The experiment used:

* 6 localization examples
* 6 held-out evaluation examples
* 2 localization examples per causal factor
* 2 evaluation examples per causal factor

For each evaluation example, the model's hidden state representation was compared with factor specific representation prototypes. The factor associated with the smallest Euclidean distance was considered the internal prediction.

### Results

| Metric                         |   Result |
| ------------------------------ | -------: |
| Localization examples          |        6 |
| Held-out evaluation examples   |        6 |
| Self-report accuracy           |     100% |
| Internal prediction accuracy   |     100% |
| Self-report/internal alignment |     100% |
| Most factor-separable layer    | Layer 34 |

The verbal self-report and activation derived prediction agreed on all six held out examples.

---

## 5. Factor Separation

The V5.2 analysis identified Layer 34 as the most factor separable layer according to the pairwise activation-distance criterion.

Across the held-out examples, the representation associated with the true factor was consistently closer to its corresponding factor prototype than to the alternative factor prototypes.

Representative examples include:

| Example | True factor | Self-report | Internal prediction |
| ------- | ----------- | ----------- | ------------------- |
| 7       | A           | A           | A                   |
| 8       | B           | B           | B                   |
| 9       | C           | C           | C                   |
| 10      | A           | A           | A                   |
| 11      | B           | B           | B                   |
| 12      | C           | C           | C                   |

Thus:

[Accuracy_{self-report} = \frac{6}{6}=100%]

[Accuracy_{internal} = \frac{6}{6}=100%]

[Alignment = \frac{6}{6}=100%]

---

## 6. Interpretation

The V5.2 results provide preliminary evidence that the model's verbal causal attributions are aligned with information encoded in its internal representations.

However, these results should not be interpreted as evidence that the model explicitly represents a human-interpretable variable such as "A" or "B" within a single localized component.

Instead, the experiment demonstrates that the activation patterns contain sufficient information to distinguish between the causal factors used in the synthetic task.

The activation-intervention experiments provide an important counterpoint. Detecting information within an activation does not necessarily demonstrate that the activation is causally necessary or sufficient for producing the model's behavior.

---

## 7. Overall Experimental Result

Taken together, the experiments support the following conclusion:

> Qwen3-4B's verbal causal self-attributions were strongly aligned with factor-discriminative information in its internal representations during the controlled V5.2 evaluation, while the causal sufficiency of those representations was not established through activation patching.

This distinction between representational alignment and causal grounding is central to interpreting the experimental results.
