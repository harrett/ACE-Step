#!/usr/bin/env python
"""Debug speaker conditioning end-to-end."""
import sys, os, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from solfa_svs.training.trainer import SolfaSVSTrainer
from solfa_svs.models.speaker_encoder import SpeakerEncoder

ckpt = sys.argv[1]              # checkpoint path
proj_path = sys.argv[2]         # speaker_projection.pt
ref_audio_a = sys.argv[3]       # reference audio A
ref_audio_b = sys.argv[4]       # reference audio B
device = sys.argv[5] if len(sys.argv) > 5 else "cpu"

print("=" * 60)
print("STEP 1: Load model from checkpoint")
print("=" * 60)
trainer = SolfaSVSTrainer.load_from_checkpoint(ckpt, map_location="cpu")
speaker_dim = trainer.hparams.get("speaker_dim", 0)
print(f"  speaker_dim from hparams: {speaker_dim}")
print(f"  solfa_dit.speaker_embedding_dim: {trainer.solfa_dit.speaker_embedding_dim}")

if hasattr(trainer.solfa_dit, 'speaker_embedder'):
    w = trainer.solfa_dit.speaker_embedder.weight
    b = trainer.solfa_dit.speaker_embedder.bias
    print(f"  speaker_embedder weight: shape={w.shape}, norm={w.norm():.4f}, "
          f"mean={w.mean():.6f}, std={w.std():.6f}")
    print(f"  speaker_embedder bias:   norm={b.norm():.4f}, "
          f"mean={b.mean():.6f}, std={b.std():.6f}")
else:
    print("  WARNING: No speaker_embedder found in solfa_dit!")

print()
print("=" * 60)
print("STEP 2: Create SpeakerEncoder + load projection")
print("=" * 60)
enc = SpeakerEncoder(output_dim=speaker_dim, device="cpu")
print(f"  Projection (before load): weight norm={enc.projection.weight.norm():.4f}")
proj_state = torch.load(proj_path, map_location="cpu", weights_only=True)
enc.projection.load_state_dict(proj_state)
print(f"  Projection (after load):  weight norm={enc.projection.weight.norm():.4f}")
enc.eval()

print()
print("=" * 60)
print("STEP 3: Extract speaker embeddings")
print("=" * 60)
try:
    emb_a = enc.encode_from_file(ref_audio_a)
    print(f"  Speaker A: shape={emb_a.shape}, device={emb_a.device}, "
          f"norm={emb_a.norm():.4f}, mean={emb_a.mean():.4f}")
except Exception as e:
    print(f"  Speaker A FAILED: {type(e).__name__}: {e}")
    emb_a = None

try:
    emb_b = enc.encode_from_file(ref_audio_b)
    print(f"  Speaker B: shape={emb_b.shape}, device={emb_b.device}, "
          f"norm={emb_b.norm():.4f}, mean={emb_b.mean():.4f}")
except Exception as e:
    print(f"  Speaker B FAILED: {type(e).__name__}: {e}")
    emb_b = None

if emb_a is not None and emb_b is not None:
    cos = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
    l2 = (emb_a - emb_b).norm().item()
    print(f"  A vs B: cosine_sim={cos:.4f}, L2_dist={l2:.4f}")

print()
print("=" * 60)
print("STEP 4: Pass through speaker_embedder (the model's projection)")
print("=" * 60)
if emb_a is not None and hasattr(trainer.solfa_dit, 'speaker_embedder'):
    embedder = trainer.solfa_dit.speaker_embedder
    with torch.no_grad():
        token_a = embedder(emb_a.float())
        print(f"  Speaker token A: shape={token_a.shape}, "
              f"norm={token_a.norm():.4f}, mean={token_a.mean():.6f}")
    if emb_b is not None:
        with torch.no_grad():
            token_b = embedder(emb_b.float())
            print(f"  Speaker token B: shape={token_b.shape}, "
                  f"norm={token_b.norm():.4f}, mean={token_b.mean():.6f}")
            cos_t = torch.nn.functional.cosine_similarity(
                token_a.view(1, -1), token_b.view(1, -1)).item()
            l2_t = (token_a - token_b).norm().item()
            print(f"  Token A vs B: cosine_sim={cos_t:.4f}, L2_dist={l2_t:.4f}")

print()
print("=" * 60)
print("STEP 5: Compare with training data embeddings")
print("=" * 60)
import json
latent_dir = "data/dcae_latents_spk"
if os.path.exists(os.path.join(latent_dir, "train.json")):
    with open(os.path.join(latent_dir, "train.json")) as f:
        meta = json.load(f)
    # Get one embedding per speaker from training data
    seen = {}
    for entry in meta:
        spk = entry.get("speaker_name", "")
        if spk in seen:
            continue
        pt_path = os.path.join(latent_dir, entry["pt_path"])
        if os.path.exists(pt_path):
            sample = torch.load(pt_path, map_location="cpu", weights_only=False)
            train_emb = sample.get("speaker_embedding")
            if train_emb is not None and train_emb.numel() > 0:
                seen[spk] = train_emb
                print(f"  Training {spk}: norm={train_emb.norm():.4f}, mean={train_emb.mean():.4f}")
    
    if emb_a is not None:
        for spk, train_emb in seen.items():
            cos = torch.nn.functional.cosine_similarity(
                emb_a.squeeze().unsqueeze(0), train_emb.unsqueeze(0)).item()
            print(f"  Infer A vs Training {spk}: cosine_sim={cos:.4f}")
    if emb_b is not None:
        for spk, train_emb in seen.items():
            cos = torch.nn.functional.cosine_similarity(
                emb_b.squeeze().unsqueeze(0), train_emb.unsqueeze(0)).item()
            print(f"  Infer B vs Training {spk}: cosine_sim={cos:.4f}")
else:
    print("  No training data found")

print("\nDone.")
