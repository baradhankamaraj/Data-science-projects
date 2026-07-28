import torch
from langchain_huggingface import HuggingFaceEmbeddings

device = "cuda" if torch.cuda.is_available() else "cpu"

embedds = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2",
    model_kwargs={
        "device": device
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print(f"Using device: {device}")