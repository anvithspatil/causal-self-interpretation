import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-4B"


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model():

    device = get_device()

    print("=" * 70)
    print("V5 MODEL LOADER")
    print("=" * 70)

    print("Model:", MODEL_NAME)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # For Apple Silicon, float16 is generally the practical
    # starting point. CPU falls back to float32.
    if device.type == "mps":
        dtype = torch.float16
    elif device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("dtype:", dtype)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=None
    )

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")
    print("Number of layers:",
          model.config.num_hidden_layers)

    print("Hidden size:",
          model.config.hidden_size)

    return model, tokenizer, device