"""
experiments/v6_loocv/extract_clip_embeddings.py

Extracts 512-dimensional CLIP embeddings from all ledger page images.
Uses openai/clip-vit-base-patch32 (smallest CLIP model, ~150MB download).

CLIP embeddings capture semantic content — table structure, handwriting style,
layout complexity — rather than just pixel statistics like the 26 visual features.
This should better identify pages that are semantically similar even if they
look visually different (fixing the 1881_1-type failures).

Output: data/visual_features/clip_embeddings.json
  {
    "1700_7": [0.123, -0.456, ...],  # 512-dim float vector
    ...
  }

Usage:
    python -m experiments.v6_loocv.extract_clip_embeddings
    
    First run downloads ~150MB model from HuggingFace (cached after that).
"""
import os
import sys
import json
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Suppress transformers warnings about slow processors
warnings.filterwarnings("ignore", message=".*slow.*processor.*")
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import BASE_DIR

IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "visual_features", "clip_embeddings.json")


def load_clip_model():
    """Load CLIP model and processor. Downloads on first run (~150MB)."""
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        # Suppress transformers warnings
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        print("[Error] Missing dependencies. Run:")
        print("  pip install transformers torch")
        sys.exit(1)

    print("[Loading] CLIP model (openai/clip-vit-base-patch32)...")
    print("          First run downloads ~150MB — cached after that.")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Use GPU if available, otherwise CPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"[OK] CLIP loaded on {device}")
    return model, processor, device


def extract_embedding(image_path: str, model, processor, device) -> list:
    """
    Extract a 512-dimensional CLIP image embedding for one image.
    Normalized to unit length (cosine similarity = dot product).
    """
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        # L2-normalize so cosine similarity = dot product
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze().cpu().numpy().tolist()


def extract_all_embeddings(image_dir: str) -> dict:
    """Extract CLIP embeddings for all PNG images in the directory."""
    print("\n" + "="*60)
    print("PHASE 4: CLIP Embedding Extraction")
    print("="*60)

    image_files = sorted(Path(image_dir).glob("*.png"))

    if not image_files:
        print(f"[Error] No PNG images found in {image_dir}")
        return {}

    print(f"\n[Found] {len(image_files)} images in {image_dir}")

    model, processor, device = load_clip_model()

    embeddings = {}
    print(f"\n[Extracting embeddings...]")

    for img_path in tqdm(image_files, desc="CLIP embedding", unit="img"):
        # Handle both "1700_7.png" and "1700_7_image.png" formats
        page_name = img_path.stem.replace("_image", "")
        try:
            emb = extract_embedding(str(img_path), model, processor, device)
            embeddings[page_name] = emb
        except Exception as e:
            print(f"\n  [Error] {page_name}: {e}")
            continue

    print(f"\n[OK] Extracted embeddings for {len(embeddings)}/{len(image_files)} pages")
    print(f"     Embedding dimension: {len(list(embeddings.values())[0])}")

    return embeddings


def save_embeddings(embeddings: dict, output_path: str):
    """Save embeddings to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(embeddings, f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[Saved] {output_path}  ({size_kb:.1f} KB)")


def print_embedding_summary(embeddings: dict):
    """Print quick summary of extracted embeddings."""
    n_pages = len(embeddings)
    dim = len(list(embeddings.values())[0])
    pages = sorted(embeddings.keys())

    print(f"\n{'='*60}")
    print(f"EMBEDDING SUMMARY")
    print(f"{'='*60}")
    print(f"  Pages:     {n_pages}")
    print(f"  Dimension: {dim}")
    print(f"  Pages:     {', '.join(pages[:5])} ... {pages[-1]}")

    # Quick sanity check: compute pairwise cosine similarities
    # (should be in range [-1, 1], most historical pages should be fairly similar)
    vecs = np.array(list(embeddings.values()))
    sim_matrix = vecs @ vecs.T
    np.fill_diagonal(sim_matrix, np.nan)

    print(f"\n  Pairwise cosine similarity stats:")
    print(f"    Mean:   {np.nanmean(sim_matrix):.4f}")
    print(f"    Std:    {np.nanstd(sim_matrix):.4f}")
    print(f"    Min:    {np.nanmin(sim_matrix):.4f}")
    print(f"    Max:    {np.nanmax(sim_matrix):.4f}")

    # Most similar pair
    sim_copy = sim_matrix.copy()
    np.fill_diagonal(sim_copy, -np.inf)
    idx = np.unravel_index(np.argmax(sim_copy), sim_copy.shape)
    pages_list = list(embeddings.keys())
    print(f"\n  Most similar pair:  {pages_list[idx[0]]} <-> {pages_list[idx[1]]}  "
          f"(sim={sim_copy[idx]:.4f})")

    # Most dissimilar pair
    np.fill_diagonal(sim_copy, np.inf)
    idx2 = np.unravel_index(np.argmin(sim_copy), sim_copy.shape)
    print(f"  Most dissimilar:    {pages_list[idx2[0]]} <-> {pages_list[idx2[1]]}  "
          f"(sim={sim_copy[idx2]:.4f})")


if __name__ == "__main__":
    embeddings = extract_all_embeddings(IMAGE_DIR)

    if embeddings:
        save_embeddings(embeddings, OUTPUT_PATH)
        print_embedding_summary(embeddings)

        print(f"\n{'='*60}")
        print("Next steps:")
        print("  4A (binary skip):   python -m experiments.v6_loocv.loocv_clip_binary --sweep")
        print("  4B (multi-class):   python -m experiments.v6_loocv.loocv_clip_multiclass --sweep")
        print("  Compare all:        python -m experiments.v6_loocv.compare_phase4")
        print(f"{'='*60}\n")