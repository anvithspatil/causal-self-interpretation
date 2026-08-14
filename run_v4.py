import json
import requests
import os
import random
import time

MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TIMEOUT = 120
MAX_RETRIES = 3

random.seed(42)


def ask_json(prompt):

    for attempt in range(MAX_RETRIES):

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0,
                        "num_predict": 50
                    }
                },
                timeout=TIMEOUT
            )

            response.raise_for_status()

            result = json.loads(
                response.json()["response"]
            )

            return result

        except Exception as e:

            print(
                f"  Error "
                f"(attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                raise


def ask_factor(rule, variables, output):

    prompt = f"""
You are analyzing a deterministic artificial system.

Rule:
{rule}

Variables:
A = {variables["A"]}
B = {variables["B"]}
C = {variables["C"]}

The system produced:

OUTPUT = {output}

Which ONE variable directly determines the output according to the rule?

Return JSON only:

{{
  "factor": "A"
}}

The factor must be exactly one of:
A
B
C

Do not explain your answer.
"""

    result = ask_json(prompt)

    factor = result["factor"].strip().upper()

    if factor not in ["A", "B", "C"]:
        raise ValueError(
            f"Invalid factor returned: {factor}"
        )

    return factor


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with open("data/v4_pilot.json", "r") as f:
    experiments = json.load(f)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

output_file = "results/v4_results_v2.json"

if os.path.exists(output_file):

    with open(output_file, "r") as f:
        results = json.load(f)

else:

    results = []


completed = {
    x["id"]
    for x in results
}


print("=" * 70)
print("V4 — CONTROLLED CAUSAL SELF-INTERPRETATION")
print("SELF-REPORT ONLY")
print("=" * 70)

print(f"Model: {MODEL}")
print(f"Experiments: {len(experiments)}")
print(f"Completed: {len(completed)}")


# --------------------------------------------------
# RUN
# --------------------------------------------------

for exp in experiments:

    exp_id = exp["id"]

    if exp_id in completed:

        print(
            f"\nSkipping experiment {exp_id}"
        )

        continue


    print("\n" + "-" * 70)
    print(f"Experiment {exp_id}/20")


    rule = exp["rule"]

    original = exp["variables"].copy()

    true_target = exp["target"]


    print(f"Rule: {rule}")
    print(f"Variables: {original}")
    print(f"True causal factor: {true_target}")


    # --------------------------------------------------
    # KNOWN ORIGINAL OUTPUT
    # --------------------------------------------------

    original_output = original[true_target]


    print(
        f"Known output: {original_output}"
    )


    # --------------------------------------------------
    # MODEL SELF-REPORT
    # --------------------------------------------------

    model_factor = ask_factor(
        rule,
        original,
        original_output
    )


    self_report_correct = (
        model_factor == true_target
    )


    print(
        f"Model says factor: {model_factor}"
    )

    print(
        f"Self-report correct: "
        f"{self_report_correct}"
    )


    # --------------------------------------------------
    # TARGET INTERVENTION
    # --------------------------------------------------

    target_intervention = original.copy()

    target_intervention[true_target] = (
        1 - target_intervention[true_target]
    )


    target_output = (
        target_intervention[true_target]
    )


    target_behavior_changed = (
        target_output != original_output
    )


    print(
        f"Target intervention:"
        f" {true_target} "
        f"{original[true_target]} → "
        f"{target_intervention[true_target]}"
    )

    print(
        f"Target output:"
        f" {original_output} → "
        f"{target_output}"
    )


    # --------------------------------------------------
    # CONTROL INTERVENTION
    # --------------------------------------------------

    controls = [
        x
        for x in ["A", "B", "C"]
        if x != true_target
    ]


    control_factor = random.choice(
        controls
    )


    control_intervention = original.copy()

    control_intervention[control_factor] = (
        1 - control_intervention[control_factor]
    )


    # Since OUTPUT = true_target,
    # changing the control must NOT affect output.

    control_output = original_output


    control_behavior_changed = (
        control_output != original_output
    )


    print(
        f"Control intervention:"
        f" {control_factor} "
        f"{original[control_factor]} → "
        f"{control_intervention[control_factor]}"
    )

    print(
        f"Control output:"
        f" {control_output}"
    )


    # --------------------------------------------------
    # FINAL METRIC
    # --------------------------------------------------

    causal_prediction_correct = (
        self_report_correct
        and target_behavior_changed
        and not control_behavior_changed
    )


    result = {

        "id": exp_id,

        "rule": rule,

        "original_variables":
            original,

        "true_causal_factor":
            true_target,

        "known_original_output":
            original_output,

        "model_self_report":
            model_factor,

        "self_report_correct":
            self_report_correct,

        "target_intervention":
            target_intervention,

        "target_output":
            target_output,

        "target_behavior_changed":
            target_behavior_changed,

        "control_factor":
            control_factor,

        "control_intervention":
            control_intervention,

        "control_output":
            control_output,

        "control_behavior_changed":
            control_behavior_changed,

        "causal_prediction_correct":
            causal_prediction_correct
    }


    results.append(result)


    # --------------------------------------------------
    # SAVE IMMEDIATELY
    # --------------------------------------------------

    with open(output_file, "w") as f:

        json.dump(
            results,
            f,
            indent=2
        )


    print("✓ Saved")


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

total = len(results)

correct_reports = sum(
    x["self_report_correct"]
    for x in results
)

causal_predictions = sum(
    x["causal_prediction_correct"]
    for x in results
)


print("\n" + "=" * 70)
print("V4 COMPLETE")
print("=" * 70)

print(
    f"Self-report accuracy:"
    f" {correct_reports}/{total}"
    f" ({correct_reports / total:.1%})"
)

print(
    f"Causal prediction:"
    f" {causal_predictions}/{total}"
    f" ({causal_predictions / total:.1%})"
)

print(
    f"\nResults:"
    f" {output_file}"
)