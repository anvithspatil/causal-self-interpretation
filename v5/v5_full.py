import json
import os
import re
import random
import statistics

import torch

from model import load_model


# ============================================================
# V5.1 CONFIG
# ============================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

LOCALIZATION_PER_FACTOR = 1
EVALUATION_PER_FACTOR = 1

# We only search the final 12 layers.
# This is a speed/experimental-design choice, not a claim
# that earlier layers cannot contain the information.
SEARCH_LAST_N_LAYERS = 12


# ============================================================
# LOAD MODEL
# ============================================================

MODEL, TOKENIZER, DEVICE = load_model()

NUM_LAYERS = MODEL.config.num_hidden_layers


if NUM_LAYERS < SEARCH_LAST_N_LAYERS:
    SEARCH_START_LAYER = 0
else:
    SEARCH_START_LAYER = (
        NUM_LAYERS - SEARCH_LAST_N_LAYERS
    )


print(
    f"\nSearching layers "
    f"{SEARCH_START_LAYER}–{NUM_LAYERS - 1}"
)


# ============================================================
# LOAD DATA
# ============================================================

with open("../data/v4_pilot.json", "r") as f:
    base_data = json.load(f)


# ============================================================
# BALANCED DATA SPLIT
# ============================================================

by_factor = {
    "A": [],
    "B": [],
    "C": []
}


for example in base_data:

    factor = example["target"]

    if factor in by_factor:
        by_factor[factor].append(example)


LOCALIZATION_DATA = []
EVALUATION_DATA = []


for factor in ["A", "B", "C"]:

    examples = by_factor[factor]

    required = (
        LOCALIZATION_PER_FACTOR
        + EVALUATION_PER_FACTOR
    )

    if len(examples) < required:

        raise RuntimeError(
            f"Need at least {required} "
            f"examples for factor {factor}, "
            f"but found {len(examples)}."
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
            + EVALUATION_PER_FACTOR
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

def build_prompt(rule, variables):

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
# TOKENIZATION
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
# OUTPUT TOKEN IDS
# ============================================================

zero_tokens = TOKENIZER.encode(
    "0",
    add_special_tokens=False
)

one_tokens = TOKENIZER.encode(
    "1",
    add_special_tokens=False
)


if len(zero_tokens) != 1:
    raise RuntimeError(
        "'0' is not a single token."
    )

if len(one_tokens) != 1:
    raise RuntimeError(
        "'1' is not a single token."
    )


ZERO_ID = zero_tokens[0]
ONE_ID = one_tokens[0]


# ============================================================
# LOGIT MARGIN
# ============================================================

def output_margin(logits):

    last_logits = logits[:, -1, :]

    return (
        last_logits[:, ONE_ID]
        - last_logits[:, ZERO_ID]
    ).item()


# ============================================================
# FORWARD PASS
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
# MODEL LAYERS
# ============================================================

if hasattr(MODEL, "model") and hasattr(
    MODEL.model,
    "layers"
):

    LAYERS = MODEL.model.layers

else:

    raise RuntimeError(
        "Could not locate transformer layers."
    )


# ============================================================
# ACTIVATION PATCH
# ============================================================

class SinglePositionPatcher:

    def __init__(
        self,
        token_index,
        source_activation
    ):

        self.token_index = token_index
        self.source_activation = source_activation


    def hook(
        self,
        module,
        inputs,
        output
    ):

        if isinstance(output, tuple):

            hidden = output[0]

            patched = hidden.clone()

            patched[
                :,
                self.token_index,
                :
            ] = self.source_activation[
                :,
                self.token_index,
                :
            ]

            return (
                patched,
                *output[1:]
            )

        else:

            patched = output.clone()

            patched[
                :,
                self.token_index,
                :
            ] = self.source_activation[
                :,
                self.token_index,
                :
            ]

            return patched


# ============================================================
# PATCH ONE REPRESENTATION
# ============================================================

@torch.no_grad()
def patch_representation(
    target_prompt,
    source_hidden,
    layer,
    token
):

    target_inputs = tokenize(
        target_prompt
    )

    # Safety check.
    if token >= target_inputs[
        "input_ids"
    ].shape[1]:

        raise RuntimeError(
            f"Token {token} is outside "
            f"target sequence."
        )


    patcher = SinglePositionPatcher(
        token,
        source_hidden
    )


    handle = LAYERS[
        layer
    ].register_forward_hook(
        patcher.hook
    )


    try:

        outputs = MODEL(
            **target_inputs,
            return_dict=True
        )

    finally:

        handle.remove()


    return output_margin(
        outputs.logits
    )


# ============================================================
# V5-A
# CHEAP REPRESENTATION LOCALIZATION
# ============================================================

print("\n" + "=" * 70)
print("V5.1-A — FAST REPRESENTATION LOCALIZATION")
print("=" * 70)


candidate_scores = {
    "A": [],
    "B": [],
    "C": []
}


for example in LOCALIZATION_DATA:

    factor = example["target"]

    original = example[
        "variables"
    ].copy()

    corrupted = original.copy()

    corrupted[factor] = (
        1 - corrupted[factor]
    )


    original_prompt = build_prompt(
        example["rule"],
        original
    )

    corrupted_prompt = build_prompt(
        example["rule"],
        corrupted
    )


    original_inputs, original_outputs = (
        forward(original_prompt)
    )

    corrupted_inputs, corrupted_outputs = (
        forward(corrupted_prompt)
    )


    # --------------------------------------------------------
    # TOKEN ALIGNMENT
    # --------------------------------------------------------

    original_ids = (
        original_inputs["input_ids"]
    )

    corrupted_ids = (
        corrupted_inputs["input_ids"]
    )


    if (
        original_ids.shape
        != corrupted_ids.shape
    ):

        raise RuntimeError(
            "Contrastive pair has "
            "different token lengths."
        )


    seq_len = (
        original_ids.shape[1]
    )


    # --------------------------------------------------------
    # ACTIVATION DIFFERENCE
    # --------------------------------------------------------

    for layer in range(
        SEARCH_START_LAYER,
        NUM_LAYERS
    ):

        clean_hidden = (
            original_outputs
            .hidden_states[layer + 1]
        )

        corrupted_hidden = (
            corrupted_outputs
            .hidden_states[layer + 1]
        )


        difference = (
            clean_hidden
            - corrupted_hidden
        )


        # Per-token L2 difference.
        token_scores = torch.norm(
            difference,
            dim=-1
        )[0]


        best_token = int(
            torch.argmax(
                token_scores
            ).item()
        )


        best_score = float(
            token_scores[
                best_token
            ].item()
        )


        candidate_scores[
            factor
        ].append({

            "example_id":
                example["id"],

            "layer":
                layer,

            "token":
                best_token,

            "activation_difference":
                best_score
        })


# ============================================================
# SELECT ONE CANDIDATE PER FACTOR
# ============================================================

candidates = {}


for factor in ["A", "B", "C"]:

    entries = candidate_scores[
        factor
    ]


    # Aggregate repeated observations
    # by layer/token pair.

    grouped = {}


    for entry in entries:

        key = (
            entry["layer"],
            entry["token"]
        )

        grouped.setdefault(
            key,
            []
        ).append(
            entry[
                "activation_difference"
            ]
        )


    ranked = []


    for (
        layer,
        token
    ), values in grouped.items():

        ranked.append({

            "layer":
                layer,

            "token":
                token,

            "mean_difference":
                statistics.mean(
                    values
                ),

            "observations":
                len(values)
        })


    ranked.sort(
        key=lambda x:
            x["mean_difference"],
        reverse=True
    )


    if not ranked:

        raise RuntimeError(
            f"No candidate found for {factor}"
        )


    candidates[factor] = ranked[0]


print("\nSelected representations:")


for factor in ["A", "B", "C"]:

    c = candidates[factor]

    print(
        f"{factor}: "
        f"layer={c['layer']}, "
        f"token={c['token']}, "
        f"score={c['mean_difference']:.4f}"
    )


# ============================================================
# SAVE LOCALIZATION
# ============================================================

with open(
    f"{RESULT_DIR}/v5_1_localization.json",
    "w"
) as f:

    json.dump(
        {
            "search_layers": [
                SEARCH_START_LAYER,
                NUM_LAYERS - 1
            ],
            "candidates": candidates,
            "raw_scores": candidate_scores
        },
        f,
        indent=2
    )


# ============================================================
# V5-B/C/D
# CROSS-PATCHING + BEHAVIOR + SELF REPORT
# ============================================================

print("\n" + "=" * 70)
print("V5.1-B/C/D — CROSS-PATCHING + BEHAVIOR + SELF-REPORT")
print("=" * 70)


final_results = []


for example in EVALUATION_DATA:

    example_id = example["id"]

    true_factor = example["target"]

    original = (
        example["variables"].copy()
    )


    # --------------------------------------------------------
    # ORIGINAL FORWARD PASS
    # --------------------------------------------------------

    original_prompt = build_prompt(
        example["rule"],
        original
    )


    original_inputs, original_outputs = (
        forward(original_prompt)
    )


    clean_margin = output_margin(
        original_outputs.logits
    )


    original_output = int(
        original[true_factor]
    )


    # --------------------------------------------------------
    # SELF REPORT
    # --------------------------------------------------------

    report = self_report(
        example["rule"],
        original,
        original_output
    )


    self_report_correct = (
        report == true_factor
    )


    # --------------------------------------------------------
    # BUILD TARGET-CONTRASTIVE PROMPT
    # --------------------------------------------------------

    target_corrupted = (
        original.copy()
    )

    target_corrupted[
        true_factor
    ] = (
        1 -
        target_corrupted[
            true_factor
        ]
    )


    target_prompt = build_prompt(
        example["rule"],
        target_corrupted
    )


    target_inputs, target_outputs = (
        forward(target_prompt)
    )


    corrupted_margin = output_margin(
        target_outputs.logits
    )


    # --------------------------------------------------------
    # GET SOURCE ACTIVATIONS
    # --------------------------------------------------------

    source_activations = {}


    for source_factor in [
        "A",
        "B",
        "C"
    ]:

        source_variables = (
            original.copy()
        )

        # Source representation is obtained
        # from the actual source factor condition.

        source_variables[
            source_factor
        ] = 1


        source_prompt = build_prompt(
            example["rule"],
            source_variables
        )


        source_inputs, source_outputs = (
            forward(source_prompt)
        )


        candidate = candidates[
            source_factor
        ]


        layer = candidate[
            "layer"
        ]

        token = candidate[
            "token"
        ]


        if token >= source_inputs[
            "input_ids"
        ].shape[1]:

            raise RuntimeError(
                f"Candidate token {token} "
                f"is invalid for source "
                f"{source_factor}."
            )


        source_activations[
            source_factor
        ] = (
            source_outputs
            .hidden_states[
                layer + 1
            ]
        )


    # --------------------------------------------------------
    # CROSS-PATCH MATRIX
    # --------------------------------------------------------

    patch_matrix = {}


    for source_factor in [
        "A",
        "B",
        "C"
    ]:

        patch_matrix[
            source_factor
        ] = {}


        candidate = candidates[
            source_factor
        ]


        layer = candidate[
            "layer"
        ]

        token = candidate[
            "token"
        ]


        patched_margin = patch_representation(
            target_prompt,
            source_activations[
                source_factor
            ],
            layer,
            token
        )


        effect = (
            patched_margin
            - corrupted_margin
        )


        patch_matrix[
            source_factor
        ] = {

            "layer":
                layer,

            "token":
                token,

            "patched_margin":
                patched_margin,

            "effect":
                effect
        }


    # --------------------------------------------------------
    # MATCHED / MISMATCHED
    # --------------------------------------------------------

    matched_effect = abs(
        patch_matrix[
            true_factor
        ]["effect"]
    )


    mismatched_effects = [

        abs(
            patch_matrix[factor][
                "effect"
            ]
        )

        for factor in [
            "A",
            "B",
            "C"
        ]

        if factor != true_factor
    ]


    mean_mismatched_effect = (
        statistics.mean(
            mismatched_effects
        )
    )


    if mean_mismatched_effect < 1e-8:

        specificity_ratio = float(
            "inf"
        )

    else:

        specificity_ratio = (
            matched_effect
            /
            mean_mismatched_effect
        )


    # --------------------------------------------------------
    # SIMPLE GROUNDING CRITERION
    # --------------------------------------------------------

    causal_grounding = (

        self_report_correct

        and

        matched_effect
        >
        mean_mismatched_effect

        and

        matched_effect > 0.05
    )


    result = {

        "id":
            example_id,

        "true_factor":
            true_factor,

        "self_report":
            report,

        "self_report_correct":
            self_report_correct,

        "clean_margin":
            clean_margin,

        "corrupted_margin":
            corrupted_margin,

        "target_corruption_effect":
            corrupted_margin
            - clean_margin,

        "patch_matrix":
            patch_matrix,

        "matched_effect":
            matched_effect,

        "mean_mismatched_effect":
            mean_mismatched_effect,

        "specificity_ratio":
            specificity_ratio,

        "causal_grounding":
            causal_grounding
    }


    final_results.append(
        result
    )


    print(
        f"Example {example_id}: "
        f"true={true_factor}, "
        f"report={report}, "
        f"matched={matched_effect:.4f}, "
        f"mismatched="
        f"{mean_mismatched_effect:.4f}, "
        f"ratio="
        f"{specificity_ratio:.2f}, "
        f"grounding="
        f"{causal_grounding}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

n = len(final_results)


self_report_accuracy = (
    sum(
        x["self_report_correct"]
        for x in final_results
    )
    /
    n
)


grounding_rate = (
    sum(
        x["causal_grounding"]
        for x in final_results
    )
    /
    n
)


ratios = [
    x["specificity_ratio"]
    for x in final_results
    if x["specificity_ratio"]
    != float("inf")
]


median_specificity = (
    statistics.median(ratios)
    if ratios
    else float("inf")
)


# ============================================================
# SAVE
# ============================================================

output = {

    "experiment":
        "V5.1",

    "model":
        "Qwen/Qwen3-4B",

    "device":
        str(DEVICE),

    "localization_n":
        len(LOCALIZATION_DATA),

    "evaluation_n":
        len(EVALUATION_DATA),

    "candidates":
        candidates,

    "self_report_accuracy":
        self_report_accuracy,

    "causal_grounding_rate":
        grounding_rate,

    "median_specificity_ratio":
        median_specificity,

    "examples":
        final_results
}


with open(
    f"{RESULT_DIR}/v5_1_results.json",
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V5.1 COMPLETE")
print("=" * 70)

print(
    f"Localization n: "
    f"{len(LOCALIZATION_DATA)}"
)

print(
    f"Evaluation n: "
    f"{len(EVALUATION_DATA)}"
)

print(
    f"Self-report accuracy: "
    f"{self_report_accuracy:.1%}"
)

print(
    f"Causal grounding rate: "
    f"{grounding_rate:.1%}"
)

print(
    f"Median specificity ratio: "
    f"{median_specificity:.3f}"
)

print(
    "\nResults:"
    f" {RESULT_DIR}/v5_1_results.json"
)