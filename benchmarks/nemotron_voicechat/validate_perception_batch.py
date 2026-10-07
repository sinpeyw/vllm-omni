# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate and time duplex perception with local real weights and speech.

The reference is the singleton method from an explicitly supplied earlier
thinker source file. Only perception weights are loaded; no LLM, download or
serving process is needed. Timings include PCM preparation, mel, Conformer,
projection and per-request cache commit, synchronized around the whole batch.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import pybase64
import soundfile as sf
import torch
from omegaconf import DictConfig
from safetensors import safe_open
from scipy.signal import resample_poly
from torch import nn
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.nemotron_voicechat import nemotron_voicechat_thinker as thinker_module
from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.perception import AudioPerceptionModule

CACHE_KEYS = ("perception_cache_last_channel", "perception_cache_last_time", "perception_cache_last_channel_len")
FLOAT_KEYS = ("duplex_frame", *CACHE_KEYS[:2])


def load_perception(model_dir: Path, device: torch.device, dtype: torch.dtype):
    cfg = json.loads((model_dir / "config.json").read_text())["model"]["stt"]["model"]["perception"]
    model = thinker_module.NemotronVoiceChatThinkerForConditionalGeneration.__new__(
        thinker_module.NemotronVoiceChatThinkerForConditionalGeneration
    )
    nn.Module.__init__(model)
    model._dtype = dtype
    model._sessions = {}
    model._duplex_previous_text_tokens = {}
    model.perception = AudioPerceptionModule(DictConfig(cfg))
    state = {}
    prefix = "stt_model.perception."
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if key.startswith(prefix):
                state[key.removeprefix(prefix)] = checkpoint.get_tensor(key)
    if not state:
        raise ValueError("Local checkpoint has no VoiceChat perception weights")
    model.perception.load_state_dict(state, strict=True)
    model.perception.to(device=device, dtype=dtype).eval()
    frontend_cfg = copy.deepcopy(model.perception.cfg.preprocessor)
    frontend_cfg.dither = 0.0
    frontend_cfg.pad_to = 0
    model._streaming_preprocessor = model.perception.from_config_dict(frontend_cfg).to(device).eval()
    return model, {"loaded_tensors": len(state), "strict_load": True}


def load_reference(source: Path):
    text = source.read_text()
    tree = ast.parse(text, filename=str(source))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NemotronVoiceChatThinkerForConditionalGeneration"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_duplex_stable_frame"
    )
    if any(isinstance(node, ast.Attribute) and node.attr == "_duplex_stable_frames" for node in ast.walk(method)):
        raise ValueError("Reference source must contain the original singleton implementation")
    namespace = vars(thinker_module).copy()
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_duplex_stable_frame"], hashlib.sha256(text.encode()).hexdigest()


def load_audio(path: Path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // divisor, sr // divisor)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Real speech fixture must be nonempty and finite")
    return np.asarray(audio, dtype="<f4")


def append(audio: np.ndarray, row: int, seq: int, *, changed=False):
    samples = np.take(audio, np.arange(1280) + (seq - 1 + row * 19) * 1280, mode="wrap")
    if changed:
        samples = -samples + 0.1
    return {
        "source_input_seq": seq,
        "payload": {
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "audio": pybase64.b64encode(samples.astype("<f4").tobytes()).decode("ascii"),
        },
    }


def distribution(samples):
    return {
        "samples": len(samples),
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.sort(samples)[math.ceil(0.95 * len(samples)) - 1]),
        "stddev_ms": float(np.std(samples)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


def timed(call, device):
    torch.accelerator.synchronize(device)
    allocated_before = torch.accelerator.memory_allocated(device)
    torch.accelerator.reset_peak_memory_stats(device)
    start = time.perf_counter_ns()
    call()
    torch.accelerator.synchronize(device)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    peak = torch.accelerator.max_memory_allocated(device)
    return elapsed, peak, peak - allocated_before


def compare(actual, expected, errors):
    assert actual["last_input_seq"] == expected["last_input_seq"]
    assert torch.equal(actual[CACHE_KEYS[2]], expected[CACHE_KEYS[2]])
    for key in FLOAT_KEYS:
        a, b = actual[key].float(), expected[key].float()
        assert a.shape == b.shape and bool(torch.isfinite(a).all())
        diff = a - b
        stats = errors.setdefault(key, {"max_abs": 0.0, "max_relative_l2": 0.0})
        stats["max_abs"] = max(stats["max_abs"], float(diff.abs().max()))
        relative = float(diff.norm() / b.norm().clamp_min(1e-12))
        stats["max_relative_l2"] = max(stats["max_relative_l2"], relative)


def clone_state(state):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state.items()}


@torch.inference_mode()
def lifecycle_checks(model, reference, audio, device):
    actual, expected = [{} for _ in range(4)], [{} for _ in range(4)]
    sequences = [0] * 4
    errors = {}
    for step in range(48):
        rows = [3, 1, 0, 2] if step % 2 else [0, 2, 1, 3]
        if step % 5 == 0:
            rows.remove(1)
        if step < 7:
            rows.remove(3)
        if step == 29:
            actual[3], expected[3] = {}, {}
            sequences[3] = 44  # A reopened request restarts its stream-local clock.
        before = {row: clone_state(actual[row]) for row in set(range(4)) - set(rows)}
        entries = []
        for row in rows:
            sequences[row] += 1
            duplex = append(audio, row, sequences[row])
            reference(model, expected[row], duplex, device)
            entries.append((actual[row], duplex))
        model._duplex_stable_frames(entries, device)
        for row in rows:
            compare(actual[row], expected[row], errors)
        for row, snapshot in before.items():
            for key in FLOAT_KEYS + (CACHE_KEYS[2],):
                if key in snapshot:
                    assert torch.equal(actual[row][key], snapshot[key])
        # A replayed append must retain the exact objects and do no cache work.
        snapshots = [dict(state) for state, _ in entries]
        model._duplex_stable_frames(entries, device)
        for (state, _), snapshot in zip(entries, snapshots):
            assert all(state[key] is snapshot[key] for key in FLOAT_KEYS + (CACHE_KEYS[2],))
    for key in CACHE_KEYS:
        assert len({state[key].untyped_storage().data_ptr() for state in actual}) == 4

    # With the batch geometry unchanged, modifying one stream must not change
    # another stream's output or cache, including in reduced precision.
    states = [{} for _ in range(4)]
    for seq in range(1, 9):
        model._duplex_stable_frames([(states[row], append(audio, row, seq)) for row in range(4)], device)
    other = [clone_state(state) for state in states]
    model._duplex_stable_frames([(states[row], append(audio, row, 9)) for row in range(4)], device)
    model._duplex_stable_frames([(other[row], append(audio, row, 9, changed=(row == 1))) for row in range(4)], device)
    for row in [0, 2, 3]:
        assert all(torch.equal(states[row][key], other[row][key]) for key in FLOAT_KEYS + (CACHE_KEYS[2],))
    assert not torch.equal(states[1]["duplex_frame"], other[1]["duplex_frame"])
    return {"independent_rows_bitwise_equal": True, "pause_replay_reorder_reopen_passed": True, "errors": errors}


@torch.inference_mode()
def benchmark(model, reference, audio, device, batch, steps, warmup):
    serial_states, batch_states = [{} for _ in range(batch)], [{} for _ in range(batch)]
    serial_times, batch_times = [], []
    serial_memory, batch_memory = [], []
    errors = {}
    for seq in range(1, steps + 1):
        rows = list(reversed(range(batch))) if seq % 2 else list(range(batch))
        packets = [(row, append(audio, row, seq)) for row in rows]

        def serial_call():
            for row, duplex in packets:
                reference(model, serial_states[row], duplex, device)

        def batch_call():
            model._duplex_stable_frames([(batch_states[row], duplex) for row, duplex in packets], device)

        # Alternate order to avoid giving one path a systematic clock/cache advantage.
        if seq % 2:
            serial_sample = timed(serial_call, device)
            batch_sample = timed(batch_call, device)
        else:
            batch_sample = timed(batch_call, device)
            serial_sample = timed(serial_call, device)
        if seq > warmup:
            serial_times.append(serial_sample[0])
            batch_times.append(batch_sample[0])
            serial_memory.append(serial_sample[1:])
            batch_memory.append(batch_sample[1:])
        for row in rows:
            compare(batch_states[row], serial_states[row], errors)
    return {
        "batch": batch,
        "serial": distribution(serial_times),
        "batched": distribution(batch_times),
        "speedup_mean": float(np.mean(serial_times) / np.mean(batch_times)),
        "errors": errors,
        "memory": {
            "serial_peak_allocated_bytes": max(peak for peak, _ in serial_memory),
            "batched_peak_allocated_bytes": max(peak for peak, _ in batch_memory),
            "serial_peak_step_extra_bytes": max(extra for _, extra in serial_memory),
            "batched_peak_step_extra_bytes": max(extra for _, extra in batch_memory),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Select a new output path: {args.output}")
    if not 0 <= args.warmup < args.steps or any(batch <= 0 for batch in args.batches) or args.repeats <= 0:
        parser.error("Require positive batches/repeats and 0 <= warmup < steps")
    torch.set_num_threads(4)
    if args.dtype == "float32":
        # Use full FP32 to separate batching/state errors from reduced-precision
        # rounding. Both reference and batched paths use these same settings.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, weights = load_perception(args.model, device, getattr(torch, args.dtype))
    reference, reference_hash = load_reference(args.reference_source)
    audio = load_audio(args.audio)
    lifecycle = lifecycle_checks(model, reference, audio, device)
    rows = []
    for repeat in range(args.repeats):
        for batch in args.batches:
            row = benchmark(model, reference, audio, device, batch, args.steps, args.warmup)
            row["repeat"] = repeat + 1
            rows.append(row)
            print(json.dumps(row), flush=True)
    # Batch shape changes can change GEMM rounding. Cross-stream isolation is
    # checked bitwise at a fixed shape; reference drift is reported separately.
    error_limit = 0.02 if args.dtype == "bfloat16" else 1e-4
    all_errors = [lifecycle["errors"], *(row["errors"] for row in rows)]
    passed = all(stat["max_relative_l2"] <= error_limit for errors in all_errors for stat in errors.values())
    result = {
        "passed": passed,
        "model": str(args.model),
        "audio": str(args.audio),
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "dtype": args.dtype,
        "gpu": current_platform.get_device_name(device.index or 0),
        "torch_version": torch.__version__,
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_omni_version": importlib.metadata.version("vllm-omni"),
        "implementation_source": str(Path(thinker_module.__file__).resolve()),
        "implementation_sha256": hashlib.sha256(Path(thinker_module.__file__).read_bytes()).hexdigest(),
        "reference_source": str(args.reference_source),
        "reference_sha256": reference_hash,
        "steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "weights": weights,
        "timing_scope": "synchronized whole-batch wall time; PCM/mel/Conformer/projection/cache commit",
        "memory_scope": "peak allocated per step; both A/B state sets retained; reset before each timed call",
        "relative_l2_limit": error_limit,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "lifecycle": lifecycle,
        "batches": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(f"passed={passed}; results={args.output}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
