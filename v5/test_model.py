import torch

from model import load_model


model, tokenizer, device = load_model()


prompt = """
You are given a deterministic artificial system.

Rule:
OUTPUT = A

Variables:
A = 1
B = 0
C = 1

What is the output?
"""


inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


with torch.no_grad():

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )


print("\n" + "=" * 70)
print("FORWARD PASS SUCCESS")
print("=" * 70)

print(
    "Number of hidden-state tensors:",
    len(outputs.hidden_states)
)

print(
    "First hidden-state shape:",
    outputs.hidden_states[0].shape
)

print(
    "Last hidden-state shape:",
    outputs.hidden_states[-1].shape
)


# --------------------------------------------------
# CHECK LOGITS
# --------------------------------------------------

last_logits = outputs.logits[:, -1, :]

next_token = torch.argmax(
    last_logits,
    dim=-1
)

decoded = tokenizer.decode(
    next_token
)

print(
    "Most likely next token:",
    repr(decoded)
)


# --------------------------------------------------
# CHECK MEMORY
# --------------------------------------------------

if device.type == "mps":

    print("\nMPS backend detected.")

elif device.type == "cuda":

    print("\nCUDA backend detected.")

else:

    print("\nRunning on CPU.")


print("\nV5 model/activation access test PASSED.")