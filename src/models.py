"""Model registry (open_clip). Keys are used in cache file names."""
from __future__ import annotations

MODELS = {
    # Model used in Mind the Gap, Table 1 (clip.load("ViT-B/16")).
    # OpenAI weights were trained with QuickGELU; the "-quickgelu" config is
    # required, the plain config silently uses GELU and lowers accuracy
    # (CIFAR10 zero-shot, 2k images: .879 GELU vs .9065 QuickGELU).
    "clip_b16": ("ViT-B-16-quickgelu", "openai"),
    # Comparison with I0T / Liu et al. numbers.
    "clip_b32": ("ViT-B-32-quickgelu", "openai"),
    # Generality check, different training objective (sigmoid loss).
    "siglip_b16": ("ViT-B-16-SigLIP", "webli"),
}


def load(key: str, device: str = "cuda"):
    import open_clip

    arch, pretrained = MODELS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(arch)
    return model, preprocess, tokenizer
