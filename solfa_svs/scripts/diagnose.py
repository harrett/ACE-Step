#!/usr/bin/env python
"""
Diagnostic script: test model output and DCAE vocoder separately.

Usage:
    python -m solfa_svs.scripts.diagnose \
        --checkpoint_path exps/solfa_svs/logs/solfa_svs/checkpoints/last.ckpt \
        --midi_path data/midi_for_ace/1084180.mid \
        --dcae_checkpoint_dir .cache/ace-step/checkpoints \
        --latent_dir data/dcae_latents \
        --output_dir outputs/diagnose \
        --device cuda:0 --bf16
"""

import argparse
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchaudio
import numpy as np


def truncate_to_max_frames(latent, f0, energy, phonemes, notes, max_frames, frame_rate):
    """Truncate latent, frame features, and notes to max_frames."""
    if max_frames is None or len(f0) <= max_frames:
        return latent, f0, energy, phonemes, notes

    max_sec = max_frames / frame_rate
    f0 = f0[:max_frames]
    energy = energy[:max_frames]
    phonemes = phonemes[:max_frames]
    if latent is not None:
        latent = latent[:, :, :max_frames]
    # Filter and clip notes to fit
    notes = [n for n in notes if n["onset_sec"] < max_sec]
    for n in notes:
        if n["offset_sec"] > max_sec:
            n["offset_sec"] = max_sec
            n["duration_sec"] = n["offset_sec"] - n["onset_sec"]
            if "offset_frame" in n:
                n["offset_frame"] = max_frames
    return latent, f0, energy, phonemes, notes


def prepare_note_tensors(notes, L, frame_rate, device, dtype):
    """Build note-level tensors from a list of note dicts."""
    total_dur = L / frame_rate if L > 0 else 1.0
    max_notes = max(len(notes), 1)
    note_phonemes = torch.zeros(1, max_notes, dtype=torch.long, device=device)
    note_pitches = torch.zeros(1, max_notes, dtype=torch.long, device=device)
    note_velocities = torch.zeros(1, max_notes, dtype=torch.long, device=device)
    note_durations = torch.zeros(1, max_notes, device=device, dtype=dtype)
    note_positions = torch.zeros(1, max_notes, device=device, dtype=dtype)
    note_mask = torch.zeros(1, max_notes, device=device, dtype=dtype)

    for j, note in enumerate(notes):
        note_phonemes[0, j] = note["phoneme_id"]
        note_pitches[0, j] = note["midi_pitch"]
        note_velocities[0, j] = note["velocity"]
        note_durations[0, j] = note["duration_sec"]
        note_positions[0, j] = note["onset_sec"] / total_dur
        note_mask[0, j] = 1.0

    return note_phonemes, note_pitches, note_velocities, note_durations, note_positions, note_mask


def run_denoising(solfa_dit, encoder_hidden_states, encoder_mask,
                  attention_mask, L, num_steps, device, dtype, seed,
                  label="", omega=0.0):
    """Run the full denoising loop and return the final latent."""
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
        retrieve_timesteps,
    )
    from diffusers.utils.torch_utils import randn_tensor
    from acestep.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    latent = randn_tensor(
        shape=(1, 8, 16, L), generator=generator, device=device, dtype=dtype,
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0,
    )
    timesteps, _ = retrieve_timesteps(
        scheduler, num_inference_steps=num_steps, device=device,
    )

    print(f"  Running {num_steps} denoising steps ({label})...")
    with torch.no_grad():
        for step_i, t in enumerate(timesteps):
            noise_pred = solfa_dit(
                hidden_states=latent,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_mask=encoder_mask,
                timestep=t.expand(1).to(dtype),
            )

            latent = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latent,
                return_dict=False,
                omega=omega,
            )[0]

            if step_i in [0, 4, 9, 29, num_steps - 1]:
                lf = latent.float()
                print(f"    Step {step_i:3d} (t={t.item():.1f}): "
                      f"mean={lf.mean():.4f}  std={lf.std():.4f}  "
                      f"min={lf.min():.4f}  max={lf.max():.4f}")

    return latent


def main():
    parser = argparse.ArgumentParser(description="SolfaSVS Diagnostics")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--midi_path", type=str, required=True)
    parser.add_argument("--dcae_checkpoint_dir", type=str, default=None)
    parser.add_argument("--latent_dir", type=str, default="data/dcae_latents",
                        help="Directory with training .pt files")
    parser.add_argument("--output_dir", type=str, default="outputs/diagnose")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output_sr", type=int, default=44100)
    parser.add_argument("--num_inference_steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_duration", type=float, default=60.0,
                        help="Maximum duration in seconds for diagnosis (default: 60)")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    LATENT_FRAME_RATE = 44100 / 512 / 8
    max_frames = int(args.max_duration * LATENT_FRAME_RATE)
    print(f"Max duration: {args.max_duration}s ({max_frames} latent frames)")

    # =========================================================================
    # Step 1: Load DCAE
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Load DCAE decoder")
    print("=" * 60)

    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

    if args.dcae_checkpoint_dir is not None:
        dcae_dir = os.path.abspath(args.dcae_checkpoint_dir)
        dcae = MusicDCAE(
            dcae_checkpoint_path=os.path.join(dcae_dir, "music_dcae_f8c8"),
            vocoder_checkpoint_path=os.path.join(dcae_dir, "music_vocoder"),
        )
    else:
        dcae = MusicDCAE()
    dcae = dcae.to(torch.float32).to(device).eval()
    dcae.requires_grad_(False)
    print("  DCAE loaded OK")

    # =========================================================================
    # Step 2: Decode training latent through DCAE
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 2: Decode a TRAINING latent through DCAE")
    print("=" * 60)

    # Find matching .pt file for the MIDI sample
    midi_basename = os.path.splitext(os.path.basename(args.midi_path))[0]
    matching_pt = os.path.join(args.latent_dir, f"{midi_basename}.pt")
    train_pt = None

    if os.path.exists(matching_pt):
        train_pt = matching_pt
        print(f"  Found MATCHING training file: {train_pt}")
    else:
        # Fallback: find any .pt file
        if os.path.isdir(args.latent_dir):
            for meta_name in ["train.json", "val.json"]:
                meta_path = os.path.join(args.latent_dir, meta_name)
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        entries = json.load(f)
                    for entry in entries[:5]:
                        pt_path = os.path.join(args.latent_dir, entry["pt_path"])
                        if os.path.exists(pt_path):
                            train_pt = pt_path
                            break
                    if train_pt:
                        break
            if train_pt is None:
                for f in os.listdir(args.latent_dir):
                    if f.endswith(".pt"):
                        train_pt = os.path.join(args.latent_dir, f)
                        break
        print(f"  No matching .pt for {midi_basename}, using: {train_pt}")

    gt_latent = None
    gt_sample = None
    if train_pt is not None:
        print(f"  Loading training sample: {train_pt}")
        gt_sample = torch.load(train_pt, map_location="cpu", weights_only=False)
        gt_latent = gt_sample["latent"]  # (8, 16, L)
        print(f"  Latent shape: {gt_latent.shape}")
        print(f"  Latent stats: mean={gt_latent.mean():.4f}  std={gt_latent.std():.4f}  "
              f"min={gt_latent.min():.4f}  max={gt_latent.max():.4f}")

        # Also print training feature stats
        gt_f0 = gt_sample["f0"].numpy()
        gt_energy = gt_sample["energy"].numpy()
        gt_phonemes = gt_sample["phonemes"].numpy()
        gt_notes = gt_sample["notes"]

        # Truncate to max_duration
        gt_latent, gt_f0, gt_energy, gt_phonemes, gt_notes = truncate_to_max_frames(
            gt_latent, gt_f0, gt_energy, gt_phonemes, gt_notes,
            max_frames, LATENT_FRAME_RATE,
        )
        # Update gt_sample in-place so later steps use truncated data
        gt_sample["f0"] = torch.from_numpy(gt_f0).float()
        gt_sample["energy"] = torch.from_numpy(gt_energy).float()
        gt_sample["phonemes"] = torch.from_numpy(gt_phonemes).long()
        gt_sample["notes"] = gt_notes

        print(f"  After truncation: latent {gt_latent.shape}, {len(gt_f0)} frames, {len(gt_notes)} notes")
        print(f"  Training features:")
        print(f"    F0: range [{gt_f0.min():.1f}, {gt_f0.max():.1f}] Hz, "
              f"voiced_frac={np.mean(gt_f0 > 0):.3f}")
        print(f"    Energy: range [{gt_energy.min():.3f}, {gt_energy.max():.3f}], "
              f"mean={gt_energy.mean():.3f}")
        print(f"    Phoneme IDs: {sorted(np.unique(gt_phonemes).tolist())}")
        print(f"    Num notes: {len(gt_notes)}")

        # Decode through DCAE
        gt_latent_4d = gt_latent.unsqueeze(0).float().to(device)
        with torch.no_grad():
            _, gt_wavs = dcae.decode(gt_latent_4d, sr=args.output_sr)
        gt_wav = gt_wavs[0].cpu().float().mean(dim=0)

        gt_out = os.path.join(args.output_dir, "step2_dcae_from_training_latent.wav")
        torchaudio.save(gt_out, gt_wav.unsqueeze(0), sample_rate=args.output_sr)
        print(f"  Saved: {gt_out} ({len(gt_wav)/args.output_sr:.2f}s)")
        print(f"  WAV stats: mean={gt_wav.mean():.6f}  std={gt_wav.std():.6f}  "
              f"max_abs={gt_wav.abs().max():.6f}")
        if gt_wav.std() > 0.01:
            print("  OK - Training latent decodes to audible audio")
        else:
            print("  WARNING: Very quiet or silent!")
    else:
        print("  SKIP - No training .pt files found")

    # =========================================================================
    # Step 3: Load model
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 3: Load trained model")
    print("=" * 60)

    from solfa_svs.training.trainer import SolfaSVSTrainer
    from solfa_svs.data.midi_parser import load_midi_file, extract_note_events
    from solfa_svs.data.f0_expression import add_expression
    from acestep.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    print("  Loading checkpoint...")
    trainer_module = SolfaSVSTrainer.load_from_checkpoint(
        args.checkpoint_path, map_location="cpu"
    )
    solfa_dit = trainer_module.solfa_dit.to(dtype).to(device).eval()
    midi_encoder = trainer_module.midi_encoder.to(dtype).to(device).eval()
    print("  Model loaded OK")

    # =========================================================================
    # Step 4: MIDI inference features
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 4: Parse MIDI and prepare inference features")
    print("=" * 60)

    print(f"  Loading MIDI: {args.midi_path}")
    features = load_midi_file(args.midi_path, frame_rate=LATENT_FRAME_RATE)
    inf_f0 = features["f0"]
    inf_energy = features["energy"]
    inf_phonemes = features["phonemes"]
    inf_notes = features["notes"]

    frame_period_ms = 1000.0 / LATENT_FRAME_RATE
    inf_f0 = add_expression(
        inf_f0, phonemes=inf_phonemes,
        frame_period_ms=frame_period_ms,
        min_note_frames=3, onset_delay_frames=2,
        ramp_in_frames=2, transition_frames=1,
    )

    # Truncate to max_duration
    _, inf_f0, inf_energy, inf_phonemes, inf_notes = truncate_to_max_frames(
        None, inf_f0, inf_energy, inf_phonemes, inf_notes,
        max_frames, LATENT_FRAME_RATE,
    )

    L_inf = len(inf_f0)
    print(f"  Inference features:")
    print(f"    L={L_inf} frames, {len(inf_notes)} notes, "
          f"duration={features['duration_sec']:.1f}s")
    print(f"    F0: range [{inf_f0.min():.1f}, {inf_f0.max():.1f}] Hz, "
          f"voiced_frac={np.mean(inf_f0 > 0):.3f}")
    print(f"    Energy: range [{inf_energy.min():.3f}, {inf_energy.max():.3f}], "
          f"mean={inf_energy.mean():.3f}")
    print(f"    Phoneme IDs: {sorted(np.unique(inf_phonemes).tolist())}")
    print(f"    Num notes: {len(inf_notes)}")

    # =========================================================================
    # Step 5: COMPARE training features vs inference features
    # =========================================================================
    if gt_sample is not None:
        print()
        print("=" * 60)
        print("STEP 5: FEATURE COMPARISON (Training .pt vs MIDI inference)")
        print("=" * 60)

        gt_f0 = gt_sample["f0"].numpy()
        gt_energy = gt_sample["energy"].numpy()
        gt_phonemes = gt_sample["phonemes"].numpy()
        gt_notes = gt_sample["notes"]

        print(f"  {'':30s} {'TRAINING':>15s} {'INFERENCE':>15s}")
        print(f"  {'Num frames':30s} {len(gt_f0):15d} {L_inf:15d}")
        print(f"  {'Num notes':30s} {len(gt_notes):15d} {len(inf_notes):15d}")

        # F0 comparison
        gt_voiced = gt_f0[gt_f0 > 0]
        inf_voiced = inf_f0[inf_f0 > 0]
        print(f"  {'F0 voiced fraction':30s} {np.mean(gt_f0 > 0):15.3f} {np.mean(inf_f0 > 0):15.3f}")
        if len(gt_voiced) > 0 and len(inf_voiced) > 0:
            print(f"  {'F0 voiced mean (Hz)':30s} {gt_voiced.mean():15.1f} {inf_voiced.mean():15.1f}")
            print(f"  {'F0 voiced std (Hz)':30s} {gt_voiced.std():15.1f} {inf_voiced.std():15.1f}")
            print(f"  {'F0 voiced min (Hz)':30s} {gt_voiced.min():15.1f} {inf_voiced.min():15.1f}")
            print(f"  {'F0 voiced max (Hz)':30s} {gt_voiced.max():15.1f} {inf_voiced.max():15.1f}")

        # Energy
        print(f"  {'Energy mean':30s} {gt_energy.mean():15.3f} {inf_energy.mean():15.3f}")
        print(f"  {'Energy std':30s} {gt_energy.std():15.3f} {inf_energy.std():15.3f}")
        print(f"  {'Energy max':30s} {gt_energy.max():15.3f} {inf_energy.max():15.3f}")

        # Phonemes
        gt_ph_dist = {int(p): np.sum(gt_phonemes == p) for p in np.unique(gt_phonemes)}
        inf_ph_dist = {int(p): np.sum(inf_phonemes == p) for p in np.unique(inf_phonemes)}
        all_phs = sorted(set(gt_ph_dist.keys()) | set(inf_ph_dist.keys()))
        print(f"\n  Phoneme frame counts:")
        print(f"  {'ID':>4s}  {'TRAINING':>10s}  {'INFERENCE':>10s}")
        for p in all_phs:
            print(f"  {p:4d}  {gt_ph_dist.get(p, 0):10d}  {inf_ph_dist.get(p, 0):10d}")

        # Note stats
        if len(gt_notes) > 0 and len(inf_notes) > 0:
            gt_durs = [n["duration_sec"] for n in gt_notes]
            inf_durs = [n["duration_sec"] for n in inf_notes]
            gt_pids = [n["phoneme_id"] for n in gt_notes]
            inf_pids = [n["phoneme_id"] for n in inf_notes]
            gt_pitches = [n["midi_pitch"] for n in gt_notes if n["midi_pitch"] > 0]
            inf_pitches = [n["midi_pitch"] for n in inf_notes if n["midi_pitch"] > 0]
            gt_vels = [n["velocity"] for n in gt_notes if n["velocity"] > 0]
            inf_vels = [n["velocity"] for n in inf_notes if n["velocity"] > 0]

            print(f"\n  Note-level stats:")
            print(f"  {'':30s} {'TRAINING':>15s} {'INFERENCE':>15s}")
            print(f"  {'Mean duration (sec)':30s} {np.mean(gt_durs):15.3f} {np.mean(inf_durs):15.3f}")
            if gt_pitches and inf_pitches:
                print(f"  {'Mean MIDI pitch':30s} {np.mean(gt_pitches):15.1f} {np.mean(inf_pitches):15.1f}")
            if gt_vels and inf_vels:
                print(f"  {'Mean velocity':30s} {np.mean(gt_vels):15.1f} {np.mean(inf_vels):15.1f}")
            print(f"  {'Unique phoneme IDs in notes':30s} {sorted(set(gt_pids))} {sorted(set(inf_pids))}")

            # Show first 10 notes
            print(f"\n  First 10 training notes:")
            for i, n in enumerate(gt_notes[:10]):
                print(f"    [{i}] ph={n['phoneme_id']:2d} pitch={n['midi_pitch']:3d} "
                      f"vel={n['velocity']:3d} dur={n['duration_sec']:.3f}s "
                      f"onset={n['onset_sec']:.3f}s")
            print(f"\n  First 10 inference notes:")
            for i, n in enumerate(inf_notes[:10]):
                print(f"    [{i}] ph={n['phoneme_id']:2d} pitch={n['midi_pitch']:3d} "
                      f"vel={n['velocity']:3d} dur={n['duration_sec']:.3f}s "
                      f"onset={n['onset_sec']:.3f}s")

    # =========================================================================
    # Step 6: Run MIDI encoder with BOTH feature sets, compare outputs
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 6: Compare encoder outputs (training vs inference features)")
    print("=" * 60)

    # Inference features → encoder
    inf_f0_t = torch.from_numpy(inf_f0).to(device=device, dtype=dtype).unsqueeze(0)
    inf_energy_t = torch.from_numpy(inf_energy).to(device=device, dtype=dtype).unsqueeze(0)
    inf_phonemes_t = torch.from_numpy(inf_phonemes).long().unsqueeze(0).to(device)
    inf_attn_mask = torch.ones(1, L_inf, device=device, dtype=dtype)

    inf_note_tensors = prepare_note_tensors(
        inf_notes, L_inf, LATENT_FRAME_RATE, device, dtype)

    with torch.no_grad():
        inf_enc_hidden, inf_enc_mask = midi_encoder(
            note_phonemes=inf_note_tensors[0],
            note_pitches=inf_note_tensors[1],
            note_velocities=inf_note_tensors[2],
            note_durations=inf_note_tensors[3],
            note_positions=inf_note_tensors[4],
            note_mask=inf_note_tensors[5],
            f0=inf_f0_t, energy=inf_energy_t,
            phonemes=inf_phonemes_t,
            attention_mask=inf_attn_mask,
        )
    inf_enc_hidden = inf_enc_hidden.to(dtype)
    print(f"  Inference encoder output:  mean={inf_enc_hidden.float().mean():.4f}  "
          f"std={inf_enc_hidden.float().std():.4f}  "
          f"min={inf_enc_hidden.float().min():.4f}  max={inf_enc_hidden.float().max():.4f}")

    # Training features → encoder (if available)
    train_enc_hidden = None
    train_enc_mask = None
    if gt_sample is not None:
        gt_f0_np = gt_sample["f0"].numpy()
        gt_energy_np = gt_sample["energy"].numpy()
        gt_phonemes_np = gt_sample["phonemes"].numpy()
        gt_notes_list = gt_sample["notes"]
        L_train = len(gt_f0_np)

        gt_f0_t = gt_sample["f0"].to(device=device, dtype=dtype).unsqueeze(0)
        gt_energy_t = gt_sample["energy"].to(device=device, dtype=dtype).unsqueeze(0)
        gt_phonemes_t = gt_sample["phonemes"].long().unsqueeze(0).to(device)
        gt_attn_mask = torch.ones(1, L_train, device=device, dtype=dtype)

        gt_note_tensors = prepare_note_tensors(
            gt_notes_list, L_train, LATENT_FRAME_RATE, device, dtype)

        with torch.no_grad():
            train_enc_hidden, train_enc_mask = midi_encoder(
                note_phonemes=gt_note_tensors[0],
                note_pitches=gt_note_tensors[1],
                note_velocities=gt_note_tensors[2],
                note_durations=gt_note_tensors[3],
                note_positions=gt_note_tensors[4],
                note_mask=gt_note_tensors[5],
                f0=gt_f0_t, energy=gt_energy_t,
                phonemes=gt_phonemes_t,
                attention_mask=gt_attn_mask,
            )
        train_enc_hidden = train_enc_hidden.to(dtype)
        print(f"  Training encoder output:   mean={train_enc_hidden.float().mean():.4f}  "
              f"std={train_enc_hidden.float().std():.4f}  "
              f"min={train_enc_hidden.float().min():.4f}  max={train_enc_hidden.float().max():.4f}")

        # Cosine similarity between the two encodings (if same length)
        if L_train == L_inf:
            cos_sim = torch.nn.functional.cosine_similarity(
                train_enc_hidden.float().reshape(-1),
                inf_enc_hidden.float().reshape(-1),
                dim=0,
            )
            print(f"  Cosine similarity (train vs inf): {cos_sim.item():.4f}")

    # =========================================================================
    # Step 7: SINGLE-STEP prediction test on training data
    # =========================================================================
    if gt_sample is not None and train_enc_hidden is not None:
        print()
        print("=" * 60)
        print("STEP 7: Single-step prediction test (model on TRAINING data)")
        print("=" * 60)

        gt_latent_dev = gt_latent.unsqueeze(0).to(device=device, dtype=dtype)
        L_latent = gt_latent.shape[2]
        # attention_mask for SolfaDiT must match latent's L, not feature L
        latent_attn_mask = torch.ones(1, L_latent, device=device, dtype=dtype)

        # Test at sigma=0.5 (midrange noise level)
        scheduler_test = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=3.0,
        )
        # Find the timestep corresponding to sigma ≈ 0.5
        # With shift=3: sigma = 3*s/(1+2*s) = 0.5 => s=0.25 => t_index=750
        test_sigmas = [0.1, 0.3, 0.5, 0.7, 0.9]

        print(f"  Testing model prediction at various sigma levels:")
        print(f"  {'sigma':>8s}  {'baseline_MSE':>14s}  {'model_MSE':>14s}  {'improvement':>14s}")

        for test_sigma in test_sigmas:
            sigma_t = torch.tensor([test_sigma], device=device, dtype=dtype)

            # Find closest timestep in scheduler
            # sigma = 3*s/(1+2*s) => s = sigma/(3-2*sigma) => t = s*1000
            s = test_sigma / (3 - 2 * test_sigma)
            test_timestep_val = test_sigma * 1000  # scheduler stores sigma*1000

            # Find nearest stored timestep
            diffs = (scheduler_test.timesteps - test_timestep_val).abs()
            nearest_idx = diffs.argmin().item()
            actual_t = scheduler_test.timesteps[nearest_idx]
            actual_sigma = scheduler_test.sigmas[nearest_idx]

            sigma_4d = actual_sigma.reshape(1, 1, 1, 1).to(device=device, dtype=dtype)

            # Create noisy latent
            torch.manual_seed(args.seed)
            noise = torch.randn_like(gt_latent_dev)
            noisy_latent = sigma_4d * noise + (1.0 - sigma_4d) * gt_latent_dev

            # Baseline: predict noisy (raw=0 → preconditioned = noisy)
            baseline_pred = noisy_latent
            baseline_mse = ((baseline_pred - gt_latent_dev) ** 2).mean().item()

            # Model prediction
            with torch.no_grad():
                raw_pred = solfa_dit(
                    hidden_states=noisy_latent,
                    attention_mask=latent_attn_mask,
                    encoder_hidden_states=train_enc_hidden,
                    encoder_hidden_mask=train_enc_mask,
                    timestep=actual_t.unsqueeze(0).to(device=device, dtype=dtype),
                )
            # Apply preconditioning (same as training)
            model_pred = raw_pred * (-sigma_4d) + noisy_latent
            model_mse = ((model_pred - gt_latent_dev) ** 2).mean().item()

            improvement = (baseline_mse - model_mse) / baseline_mse * 100
            print(f"  {actual_sigma.item():8.3f}  {baseline_mse:14.6f}  {model_mse:14.6f}  "
                  f"{improvement:+13.1f}%")

        print()
        print("  If model_MSE ≈ baseline_MSE → model has NOT learned (outputs ≈ 0)")
        print("  If model_MSE << baseline_MSE → model HAS learned to predict")

    # =========================================================================
    # Step 8: Full denoising with TRAINING features as conditioning
    # =========================================================================
    if gt_sample is not None and train_enc_hidden is not None:
        print()
        print("=" * 60)
        print("STEP 8: Full denoising with TRAINING features (not MIDI)")
        print("=" * 60)

        L_latent_8 = gt_latent.shape[2]
        latent_attn_mask_8 = torch.ones(1, L_latent_8, device=device, dtype=dtype)

        train_latent = run_denoising(
            solfa_dit, train_enc_hidden, train_enc_mask,
            latent_attn_mask_8, L_latent_8,
            num_steps=args.num_inference_steps,
            device=device, dtype=dtype, seed=args.seed,
            label="training features, no CFG, omega=0",
            omega=0.0,
        )

        tl = train_latent.float()
        print(f"\n  Train-conditioned latent stats:")
        print(f"    mean={tl.mean():.4f}  std={tl.std():.4f}  "
              f"min={tl.min():.4f}  max={tl.max():.4f}")
        print(f"  Ground truth latent stats:")
        print(f"    mean={gt_latent.mean():.4f}  std={gt_latent.std():.4f}  "
              f"min={gt_latent.min():.4f}  max={gt_latent.max():.4f}")

        with torch.no_grad():
            _, train_wavs = dcae.decode(train_latent.float(), sr=args.output_sr)
        train_wav = train_wavs[0].cpu().float().mean(dim=0)

        train_out = os.path.join(args.output_dir, "step8_denoise_with_training_features.wav")
        torchaudio.save(train_out, train_wav.unsqueeze(0), sample_rate=args.output_sr)
        print(f"  Saved: {train_out} ({len(train_wav)/args.output_sr:.2f}s)")
        print(f"  WAV stats: mean={train_wav.mean():.6f}  std={train_wav.std():.6f}  "
              f"max_abs={train_wav.abs().max():.6f}")

    # =========================================================================
    # Step 9: Full denoising with MIDI features (same as original test)
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 9: Full denoising with MIDI inference features")
    print("=" * 60)

    midi_latent = run_denoising(
        solfa_dit, inf_enc_hidden, inf_enc_mask,
        inf_attn_mask, L_inf,
        num_steps=args.num_inference_steps,
        device=device, dtype=dtype, seed=args.seed,
        label="MIDI features, no CFG, omega=0",
        omega=0.0,
    )

    ml = midi_latent.float()
    print(f"\n  MIDI-conditioned latent stats:")
    print(f"    mean={ml.mean():.4f}  std={ml.std():.4f}  "
          f"min={ml.min():.4f}  max={ml.max():.4f}")

    with torch.no_grad():
        _, midi_wavs = dcae.decode(midi_latent.float(), sr=args.output_sr)
    midi_wav = midi_wavs[0].cpu().float().mean(dim=0)

    midi_out = os.path.join(args.output_dir, "step9_denoise_with_midi_features.wav")
    torchaudio.save(midi_out, midi_wav.unsqueeze(0), sample_rate=args.output_sr)
    print(f"  Saved: {midi_out} ({len(midi_wav)/args.output_sr:.2f}s)")
    print(f"  WAV stats: mean={midi_wav.mean():.6f}  std={midi_wav.std():.6f}  "
          f"max_abs={midi_wav.abs().max():.6f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print()
    print("KEY OUTPUTS:")
    print("  step2  — DCAE from training latent (should be good audio)")
    print("  step8  — Full denoising with TRAINING .pt features")
    print("  step9  — Full denoising with MIDI-inferred features")
    print()
    print("INTERPRETATION:")
    print("  Step 7 shows if the model has learned to predict at all.")
    print("    - improvement > 0% means model is learning")
    print("    - improvement ≈ 0% means model outputs ~zero (not trained enough)")
    print()
    print("  Step 8 vs Step 9 shows if features are the issue.")
    print("    - step8=audio, step9=noise → MIDI features don't match training")
    print("    - step8=noise, step9=noise → model hasn't learned (training issue)")
    print()
    print("  Step 5+6 show detailed feature comparison.")
    print("    - Large differences in F0/energy/phonemes/notes → feature mismatch")


if __name__ == "__main__":
    main()
