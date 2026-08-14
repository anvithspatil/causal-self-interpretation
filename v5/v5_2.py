import json
import os
import re
import random
import statistics

import torch

from model import load_model


# ============================================================
# CONFIG
# ============================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

LOCALIZATION_PER_FACTOR = 2
EVALUATION_PER_FACTOR = 2

# Keep the same final-layer search region used in V5.1
SEARCH_LAST_N_LAYERS = 12


# ============================================================
# LOAD MODEL
# ============================================================

MODEL, TOKENIZER, DEVICE = load_model()

NUM_LAYERS = MODEL.config.num_hidden_layers

SEARCH_START_LAYER = max(
    0,
    NUM_LAYERS - SEARCH_LAST_N_LAYERS
)

print("=" * 70)
print("V5.2 — REPRESENTATION / SELF-REPORT ALIGNMENT")
print("=" * 70)

print(
    f"Model: Qwen/Qwen3-4B"
)

print(
    f"Device: {DEVICE}"
)

print(
    f"Searching layers "
    f"{SEARCH_START_LAYER}–{NUM_LAYERS - 1}"
)


# ============================================================
# LOAD DATA
# ============================================================

with open(
    "../data/v4_pilot.json",
    "r"
) as f:

    base_data = json.load(f)


# ============================================================
# BALANCED SPLIT
# ============================================================

by_factor = {
    "A": [],
    "B": [],
    "C": []
}


for example in base_data:

    factor = example["target"]

    if factor in by_factor:
        by_factor[factor].append(
            example
        )


LOCALIZATION_DATA = []
EVALUATION_DATA = []


for factor in ["A", "B", "C"]:

    examples = by_factor[factor]

    required = (
        LOCALIZATION_PER_FACTOR
        +
        EVALUATION_PER_FACTOR
    )

    if len(examples) < required:

        raise RuntimeError(
            f"Not enough examples for "
            f"factor {factor}"
        )

    LOCALIZATION_DATA.extend(
        examples[
            :LOCALIZATION_PER_FACTOR
        ]
    )

    EVALUATION_DATA.extend(
        examples[
            LOCALIZATION_PER_FACTOR:
            LOCALIZATION_PER_FACTOR
            +
            EVALUATION_PER_FACTOR
        ]
    )


print(
    f"Localization examples: "
    f"{len(LOCALIZATION_DATA)}"
)

print(
    f"Evaluation examples: "
    f"{len(EVALUATION_DATA)}"
)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    rule,
    variables
):

    return f"""You are given a deterministic artificial system.

Rule:
{rule}

Variables:
A = {variables["A"]}
B = {variables["B"]}
C = {variables["C"]}

The system follows the rule exactly.

What is the output?
"""


# ============================================================
# TOKENIZE
# ============================================================

def tokenize(prompt):

    encoded = TOKENIZER(
        prompt,
        return_tensors="pt"
    )

    return {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }


# ============================================================
# FORWARD
# ============================================================

@torch.no_grad()
def forward(prompt):

    inputs = tokenize(prompt)

    outputs = MODEL(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

    return inputs, outputs


# ============================================================
# SELF REPORT
# ============================================================

@torch.no_grad()
def self_report(
    rule,
    variables,
    output
):

    prompt = f"""You are analyzing a deterministic artificial system.

Rule:
{rule}

Variables:
A = {variables["A"]}
B = {variables["B"]}
C = {variables["C"]}

The system produced:

OUTPUT = {output}

Which ONE variable directly determines the output according to the rule?

Return ONLY one of:

A
B
C
"""

    inputs = tokenize(prompt)

    generated = MODEL.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False
    )

    new_tokens = generated[
        0,
        inputs["input_ids"].shape[1]:
    ]

    text = TOKENIZER.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip().upper()

    match = re.search(
        r"\b([ABC])\b",
        text
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


# ============================================================
# ACTIVATION SIGNATURE
# ============================================================

def activation_signature(
    hidden_states
):

    """
    Creates a compact representation for each layer.

    We use the mean activation across token positions.
    This avoids choosing a single token position and
    gives us a whole-sequence representation.
    """

    signatures = {}

    for layer in range(
        SEARCH_START_LAYER,
        NUM_LAYERS
    ):

        hidden = hidden_states[
            layer + 1
        ]

        # [batch, sequence, hidden]
        signature = hidden.mean(
            dim=1
        )[0]

        signatures[layer] = (
            signature.detach()
            .float()
            .cpu()
        )

    return signatures


# ============================================================
# LOCALIZATION
# ============================================================

print("\n" + "=" * 70)
print("V5.2-A — BUILDING FACTOR SIGNATURES")
print("=" * 70)


factor_signatures = {
    "A": [],
    "B": [],
    "C": []
}


for example in LOCALIZATION_DATA:

    factor = example["target"]

    variables = (
        example["variables"].copy()
    )

    prompt = build_prompt(
        example["rule"],
        variables
    )

    _, outputs = forward(
        prompt
    )

    signatures = activation_signature(
        outputs.hidden_states
    )

    factor_signatures[
        factor
    ].append(signatures)


# ============================================================
# MEAN SIGNATURE PER FACTOR
# ============================================================

factor_means = {}


for factor in ["A", "B", "C"]:

    factor_means[factor] = {}

    for layer in range(
        SEARCH_START_LAYER,
        NUM_LAYERS
    ):

        vectors = [
            x[layer]
            for x in factor_signatures[
                factor
            ]
        ]

        factor_means[
            factor
        ][layer] = torch.stack(
            vectors
        ).mean(
            dim=0
        )


# ============================================================
# FACTOR SEPARABILITY
# ============================================================

def euclidean_distance(a, b):

    return torch.norm(
        a - b
    ).item()


separability = []


for layer in range(
    SEARCH_START_LAYER,
    NUM_LAYERS
):

    ab = euclidean_distance(
        factor_means["A"][layer],
        factor_means["B"][layer]
    )

    ac = euclidean_distance(
        factor_means["A"][layer],
        factor_means["C"][layer]
    )

    bc = euclidean_distance(
        factor_means["B"][layer],
        factor_means["C"][layer]
    )

    mean_distance = statistics.mean([
        ab,
        ac,
        bc
    ])

    separability.append({
        "layer": layer,
        "A_B": ab,
        "A_C": ac,
        "B_C": bc,
        "mean_pairwise_distance":
            mean_distance
    })


separability.sort(
    key=lambda x:
        x["mean_pairwise_distance"],
    reverse=True
)


best_layer = separability[0]["layer"]


print(
    f"\nMost factor-separable layer: "
    f"{best_layer}"
)

print(
    f"Mean pairwise distance: "
    f"{separability[0]['mean_pairwise_distance']:.4f}"
)


# ============================================================
# V5.2-B — EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("V5.2-B — SELF-REPORT / INTERNAL ALIGNMENT")
print("=" * 70)


evaluation_results = []


for example in EVALUATION_DATA:

    true_factor = example["target"]

    variables = (
        example["variables"].copy()
    )

    output = int(
        variables[true_factor]
    )


    prompt = build_prompt(
        example["rule"],
        variables
    )


    _, outputs = forward(
        prompt
    )


    signatures = activation_signature(
        outputs.hidden_states
    )


    model_report = self_report(
        example["rule"],
        variables,
        output
    )


    # --------------------------------------------------------
    # Distances to factor prototypes
    # --------------------------------------------------------

    distances = {}


    current = signatures[
        best_layer
    ]


    for factor in [
        "A",
        "B",
        "C"
    ]:

        distances[factor] = (
            euclidean_distance(
                current,
                factor_means[
                    factor
                ][best_layer]
            )
        )


    predicted_internal_factor = min(
        distances,
        key=distances.get
    )


    internal_prediction_correct = (
        predicted_internal_factor
        ==
        true_factor
    )


    report_internal_alignment = (
        model_report
        ==
        predicted_internal_factor
    )


    result = {

        "id":
            example["id"],

        "true_factor":
            true_factor,

        "self_report":
            model_report,

        "self_report_correct":
            model_report
            ==
            true_factor,

        "internal_prediction":
            predicted_internal_factor,

        "internal_prediction_correct":
            internal_prediction_correct,

        "self_report_internal_alignment":
            report_internal_alignment,

        "layer":
            best_layer,

        "distances":
            distances
    }


    evaluation_results.append(
        result
    )


    print(
        f"Example {example['id']}: "
        f"true={true_factor}, "
        f"report={model_report}, "
        f"internal={predicted_internal_factor}, "
        f"aligned="
        f"{report_internal_alignment}"
    )


# ============================================================
# SUMMARY
# ============================================================

n = len(
    evaluation_results
)


self_report_accuracy = (
    sum(
        x["self_report_correct"]
        for x in evaluation_results
    )
    /
    n
)


internal_accuracy = (
    sum(
        x[
            "internal_prediction_correct"
        ]
        for x in evaluation_results
    )
    /
    n
)


alignment_rate = (
    sum(
        x[
            "self_report_internal_alignment"
        ]
        for x in evaluation_results
    )
    /
    n
)


# ============================================================
# SAVE
# ============================================================

output = {

    "experiment":
        "V5.2",

    "model":
        "Qwen/Qwen3-4B",

    "device":
        str(DEVICE),

    "localization_n":
        len(LOCALIZATION_DATA),

    "evaluation_n":
        len(EVALUATION_DATA),

    "best_layer":
        best_layer,

    "layer_separability":
        separability,

    "self_report_accuracy":
        self_report_accuracy,

    "internal_prediction_accuracy":
        internal_accuracy,

    "self_report_internal_alignment":
        alignment_rate,

    "examples":
        evaluation_results
}


with open(
    f"{RESULT_DIR}/v5_2_results.json",
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("V5.2 COMPLETE")
print("=" * 70)

print(
    f"Self-report accuracy: "
    f"{self_report_accuracy:.1%}"
)

print(
    f"Internal prediction accuracy: "
    f"{internal_accuracy:.1%}"
)

print(
    f"Self-report/internal alignment: "
    f"{alignment_rate:.1%}"
)

print(
    f"Best layer: "
    f"{best_layer}"
)

print(
    "\nResults:"
    " results/v5_2_results.json"
)