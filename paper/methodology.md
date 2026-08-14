## 1. Model

All experiments were conducted using Qwen3-4B with direct access to the model's hidden states.
The model contains 36 transformer layers with a hidden dimension of 2560.
Experiments were executed locally using the MPS backend.

## 2. Synthetic Causal Environment

We use deterministic artificial systems containing three binary variables:

- A

- B

- C

Each task specifies a rule determining the output. For example:

    OUTPUT = A

The controlled environment allows the true causal variable to be known exactly.

## 3. Contrastive Evaluation

For a given target variable, we construct contrastive conditions by changing that variable while keeping the other variables fixed.

For example:

    A = 0, B = 1, C = 1

versus

    A = 1, B = 1, C = 1

This allows changes in model representations to be associated with a specific manipulated variable.

## 4. Behavioral Self-Report

The model is asked which variable directly determines the output.

Its response is restricted to:

    A

    B

    C

The response is compared with the known causal variable.

## 5. Hidden-State Analysis

Hidden states are extracted from the transformer layers.
For representation analysis, we compare activation patterns across contrastive conditions.

V5.2 uses activation signatures constructed from the mean hidden
representation across token positions.

## 6. Representation Classification

For each causal factor, a prototype representation is constructed from localization examples.

For held-out examples, the representation is compared against the factor prototypes using Euclidean distance.

The factor with the smallest distance is treated as the internally predicted factor.

## 7. Representation–Self-Report Alignment

The internally predicted factor is compared with the model's verbal self-report.

An evaluation example is considered aligned when:

    internal prediction = verbal self-report

This measures representational alignment rather than causal sufficiency.

## 8. Activation Intervention

We additionally investigated activation patching.
Representations identified during localization were transferred between contrastive conditions to test whether changing the representation altered the model's output.

These experiments were treated as a causal test.

The intervention results were not interpreted as definitive evidence of causal grounding because the localized representations did not produce sufficiently specific behavioral restoration.

## 9. Evaluation Protocol

The final V5.2 experiment used:

- 6 localization examples

- 6 held-out evaluation examples

- 2 localization examples per causal factor

- 2 evaluation examples per causal factor

The evaluation set was not used to construct the factor prototypes.