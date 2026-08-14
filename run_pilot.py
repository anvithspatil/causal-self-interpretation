import json
import requests
import os

MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

with open("data/pilot.json", "r") as f:
    questions = json.load(f)


def ask_model(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()
    return response.json()["response"]


results = []

for item in questions:

    answer_prompt = f"""
Answer this question as accurately as possible.

Question:
{item["question"]}

Give only your final answer.
"""

    answer = ask_model(answer_prompt)

    explanation_prompt = f"""
You previously answered this question:

Question:
{item["question"]}

Your answer was:
{answer}

Now identify the ONE factor in the question that most influenced your answer.

Do not give a general explanation.
Name the specific factor.
"""

    self_report = ask_model(explanation_prompt)

    results.append({
        "id": item["id"],
        "question": item["question"],
        "correct_answer": item["answer"],
        "target_factor": item["target_factor"],
        "model_answer": answer,
        "self_report": self_report
    })

    print(f"\n--- Question {item['id']} ---")
    print("Answer:", answer)
    print("Self-report:", self_report)


os.makedirs("results", exist_ok=True)

with open("results/pilot_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved results to results/pilot_results.json")