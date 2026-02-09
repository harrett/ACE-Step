"""
Training Dataset: Loads pre-encoded DCAE latent files for diffusion training.
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class SolfaDataset(Dataset):
    """
    Dataset that loads pre-encoded .pt files containing:
    - latent: (8, 16, L) DCAE-encoded target
    - latent_length: int valid frames
    - notes: List[Dict] note events
    - f0: (L,) F0 in Hz at latent frame rate
    - energy: (L,) energy at latent frame rate
    - phonemes: (L,) phoneme IDs at latent frame rate
    """

    def __init__(
        self,
        metadata_path: str,
        latent_dir: str,
        max_length: Optional[int] = None,
    ):
        """
        Args:
            metadata_path: Path to JSON file with list of sample entries
            latent_dir: Directory containing .pt files
            max_length: Maximum latent sequence length (crops longer sequences)
        """
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.latent_dir = latent_dir
        self.max_length = max_length

        # Filter out missing files
        valid = []
        for entry in self.metadata:
            pt_path = os.path.join(latent_dir, entry["pt_path"])
            if os.path.exists(pt_path):
                valid.append(entry)
        self.metadata = valid

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.metadata[idx]
        pt_path = os.path.join(self.latent_dir, entry["pt_path"])
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)

        latent = sample["latent"]              # (8, 16, L_tensor)
        latent_length = sample["latent_length"]
        notes = sample["notes"]                # List[Dict]
        f0 = sample["f0"]                      # (L_feat,)
        energy = sample["energy"]              # (L_feat,)
        phonemes = sample["phonemes"]          # (L_feat,)

        # Use the minimum of tensor width and feature length as canonical L.
        # DCAE may pad the latent tensor slightly beyond latent_length.
        L = min(latent.shape[2], len(f0))
        latent = latent[:, :, :L]
        f0 = f0[:L]
        energy = energy[:L]
        phonemes = phonemes[:L]

        # Crop if too long
        if self.max_length is not None and L > self.max_length:
            latent = latent[:, :, :self.max_length]
            latent_length = min(latent_length, self.max_length)
            f0 = f0[:self.max_length]
            energy = energy[:self.max_length]
            phonemes = phonemes[:self.max_length]
            L = self.max_length

            # Filter notes to fit within cropped length
            frame_rate = 44100 / 512 / 8
            max_sec = L / frame_rate
            notes = [n for n in notes if n["onset_sec"] < max_sec]
            for n in notes:
                n["offset_sec"] = min(n["offset_sec"], max_sec)
                n["offset_frame"] = min(n["offset_frame"], L)
                n["duration_sec"] = n["offset_sec"] - n["onset_sec"]

        return {
            "latent": latent,
            "latent_length": latent_length,
            "notes": notes,
            "f0": f0,
            "energy": energy,
            "phonemes": phonemes,
            "L": L,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for SolfaDataset.

    Pads latents, frame features, and note sequences to batch maximum.
    Creates attention masks for valid positions.
    """
    batch_size = len(batch)

    # Find max lengths
    max_L = max(item["L"] for item in batch)
    max_notes = max(len(item["notes"]) for item in batch)
    max_notes = max(max_notes, 1)  # at least 1 note slot

    # Pad latents: (B, 8, 16, max_L)
    latents = torch.zeros(batch_size, 8, 16, max_L)
    attention_mask = torch.zeros(batch_size, max_L)
    latent_lengths = torch.zeros(batch_size, dtype=torch.long)

    # Frame features: (B, max_L)
    f0 = torch.zeros(batch_size, max_L)
    energy = torch.zeros(batch_size, max_L)
    phonemes = torch.zeros(batch_size, max_L, dtype=torch.long)

    # Note features
    note_phonemes = torch.zeros(batch_size, max_notes, dtype=torch.long)
    note_pitches = torch.zeros(batch_size, max_notes, dtype=torch.long)
    note_velocities = torch.zeros(batch_size, max_notes, dtype=torch.long)
    note_durations = torch.zeros(batch_size, max_notes)
    note_positions = torch.zeros(batch_size, max_notes)
    note_mask = torch.zeros(batch_size, max_notes)

    for i, item in enumerate(batch):
        L = item["L"]
        latents[i, :, :, :L] = item["latent"][:, :, :L]
        attention_mask[i, :L] = 1.0
        latent_lengths[i] = item["latent_length"]

        f0[i, :L] = item["f0"][:L]
        energy[i, :L] = item["energy"][:L]
        phonemes[i, :L] = item["phonemes"][:L]

        notes = item["notes"]
        total_dur = L / (44100 / 512 / 8) if L > 0 else 1.0

        for j, note in enumerate(notes):
            if j >= max_notes:
                break
            note_phonemes[i, j] = note["phoneme_id"]
            note_pitches[i, j] = note["midi_pitch"]
            note_velocities[i, j] = note["velocity"]
            note_durations[i, j] = note["duration_sec"]
            note_positions[i, j] = note["onset_sec"] / total_dur
            note_mask[i, j] = 1.0

    return {
        "latent": latents,
        "attention_mask": attention_mask,
        "latent_lengths": latent_lengths,
        "f0": f0,
        "energy": energy,
        "phonemes": phonemes,
        "note_phonemes": note_phonemes,
        "note_pitches": note_pitches,
        "note_velocities": note_velocities,
        "note_durations": note_durations,
        "note_positions": note_positions,
        "note_mask": note_mask,
    }
