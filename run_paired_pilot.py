import json
import requests
import os
import random
import time

MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TIMEOUT = 300
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
                        "num_predict": 100
                    }
                },
                timeout=TIMEOUT
            )

            response.raise_for_status()

            return json.loads(
                response.json()["response"]
            )

        except Exception as e:
            print(f"  Error: {e}")

            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
            else:
                raise


def randomize_options(options, correct):

    pairs = [
        (option, option.lower() == correct.lower())
        for option in options
    ]

    random.shuffle(pairs)

    randomized = [x[0] for x in pairs]

    correct_index = next(
        i for i, x in enumerate(pairs)
        if x[1]
    )

    return randomized, correct_index


def answer_question(question, options):

    option_text = "\n".join(
        f"{i + 1}. {option}"
        for i, option in enumerate(options)
    )

    prompt = f"""
Answer this multiple-choice question.

Question:
{question}

Options:
{option_text}

Return JSON only:

{{
  "answer": 1
}}

The answer must be the number of the selected option.
"""

    result = ask_json(prompt)

    return int(result["answer"])


def get_self_report(
    question,
    selected_option,
    factor_options
):

    factors = "\n".join(
        f"{i + 1}. {factor}"
        for i, factor in enumerate(factor_options)
    )

    prompt = f"""
You answered this question:

Question:
{question}

Your selected answer:
{selected_option}

Which ONE factor from the list below most influenced your answer?

Possible factors:

{factors}

Return JSON only:

{{
  "factor": 1
}}

Choose exactly one factor number.
Do not explain your reasoning.
"""

    result = ask_json(prompt)

    return int(result["factor"])


def build_factor_options(target):

    distractors = [
    "wording of the question",
    "names of the objects",
    "order of the information",
    "answer choices",
    "general background knowledge",
    "sentence structure",
    "target factor"
]
    

    available = [
        x for x in distractors
        if x.lower() != target.lower()
    ]

    selected = random.sample(available, 3)

    factors = [target] + selected

    random.shuffle(factors)

    return factors


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with open("data/v3_pilot.json", "r") as f:
    questions = json.load(f)


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

output_file = "results/v3_results.json"

if os.path.exists(output_file):

    with open(output_file, "r") as f:
        results = json.load(f)

else:
    results = []


completed_ids = {
    r["id"]
    for r in results
}


print("=" * 70)
print("CAUSAL SELF-INTERPRETATION — EXPERIMENT 1A v2")
print("=" * 70)

print(f"Model: {MODEL}")
print(f"Experiments: {len(questions)}")
print(f"Already completed: {len(completed_ids)}")


# --------------------------------------------------
# RUN
# --------------------------------------------------

for item in questions:

    experiment_id = item["id"]

    if experiment_id in completed_ids:
        print(f"\nSkipping {experiment_id} — already completed.")
        continue

    print("\n" + "-" * 70)
    print(f"Experiment {experiment_id}/20")
    print(f"Target: {item['target_factor']}")

    original = item["original"]
    intervention = item["intervention"]

    # ----------------------------------------------
    # SAME FACTOR CANDIDATES FOR BOTH VERSIONS
    # ----------------------------------------------

    factor_options = build_factor_options(
        item["target_factor"]
    )

    target_factor_index = (
        factor_options.index(
            item["target_factor"]
        ) + 1
    )

    print("Factor options:")
    for i, factor in enumerate(factor_options):
        print(f"  {i + 1}. {factor}")

    # ----------------------------------------------
    # RANDOMIZE ORIGINAL OPTIONS
    # ----------------------------------------------

    original_options, original_correct_index = (
        randomize_options(
            original["options"],
            original["correct"]
        )
    )

    print("\nOriginal:")

    original_answer = answer_question(
        original["question"],
        original_options
    )

    original_selected = (
        original_options[original_answer - 1]
    )

    original_factor = get_self_report(
        original["question"],
        original_selected,
        factor_options
    )

    original_correct = (
        original_answer == original_correct_index + 1
    )

    # ----------------------------------------------
    # RANDOMIZE INTERVENTION OPTIONS
    # ----------------------------------------------

    intervention_options, intervention_correct_index = (
        randomize_options(
            intervention["options"],
            intervention["correct"]
        )
    )

    print("Intervention:")

    intervention_answer = answer_question(
        intervention["question"],
        intervention_options
    )

    intervention_selected = (
        intervention_options[
            intervention_answer - 1
        ]
    )

    intervention_factor = get_self_report(
        intervention["question"],
        intervention_selected,
        factor_options
    )

    intervention_correct = (
        intervention_answer
        == intervention_correct_index + 1
    )

    # ----------------------------------------------
    # EVALUATION
    # ----------------------------------------------

    answer_changed = (
        original_selected.lower()
        != intervention_selected.lower()
    )

    original_claimed_target = (
        original_factor == target_factor_index
    )

    intervention_claimed_target = (
        intervention_factor == target_factor_index
    )

    self_report_consistent = (
        original_factor == intervention_factor
    )

    result = {

        "id": experiment_id,

        "category": item["category"],

        "target_factor": item["target_factor"],

        "factor_options": factor_options,

        "target_factor_index": target_factor_index,

        "original": {
            "question": original["question"],
            "options": original_options,
            "correct_option_index":
                original_correct_index + 1,
            "model_option_index":
                original_answer,
            "model_answer":
                original_selected,
            "correct":
                original_correct,
            "self_report_index":
                original_factor,
            "self_report":
                factor_options[
                    original_factor - 1
                ],
            "self_report_identified_target":
                original_claimed_target
        },

        "intervention": {
            "question": intervention["question"],
            "options": intervention_options,
            "correct_option_index":
                intervention_correct_index + 1,
            "model_option_index":
                intervention_answer,
            "model_answer":
                intervention_selected,
            "correct":
                intervention_correct,
            "self_report_index":
                intervention_factor,
            "self_report":
                factor_options[
                    intervention_factor - 1
                ],
            "self_report_identified_target":
                intervention_claimed_target
        },

        "answer_changed":
            answer_changed,

        "self_report_consistent":
            self_report_consistent
    }

    results.append(result)

    # SAVE AFTER EVERY EXPERIMENT
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"  Original: {original_selected}"
        f" | Correct: {original_correct}"
    )

    print(
        f"  Original report: "
        f"{factor_options[original_factor - 1]}"
    )

    print(
        f"  Intervention: {intervention_selected}"
        f" | Correct: {intervention_correct}"
    )

    print(
        f"  Intervention report: "
        f"{factor_options[intervention_factor - 1]}"
    )

    print(
        f"  Answer changed: {answer_changed}"
    )

    print(
        f"  Self-report consistent: "
        f"{self_report_consistent}"
    )

    print("  ✓ Saved")


print("\n" + "=" * 70)
print("EXPERIMENT 1A v2 COMPLETE")
print("=" * 70)

print(f"Results saved to:")
print(output_file)