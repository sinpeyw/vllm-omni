# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Time whole-batch VoiceChat perception against an explicit singleton source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import torch
from vllm.platforms import current_platform

from benchmarks.nemotron_voicechat.checkpoint_metadata import checkpoint_identity
from benchmarks.nemotron_voicechat.validate_perception_batch import (
    CACHE_KEYS,
    append,
    distribution,
    load_audio,
    load_perception,
    load_reference,
    thinker_module,
    timed,
)


@torch.inference_mode()
def benchmark(model, reference, audio, device, batch, steps, warmup):
    serial_states, batch_states = [{} for _ in range(batch)], [{} for _ in range(batch)]
    samples = {"serial": [], "batched": []}
    memory = {"serial": [], "batched": []}
    for sequence in range(1, steps + 1):
        rows = list(range(batch)) if sequence % 2 == 0 else list(reversed(range(batch)))
        packets = [(row, append(audio, row, sequence)) for row in rows]

        def serial():
            for row, packet in packets:
                reference(model, serial_states[row], packet, device)

        def batched():
            model._duplex_stable_frames([(batch_states[row], packet) for row, packet in packets], device)

        calls = [("serial", serial), ("batched", batched)]
        if sequence % 2 == 0:
            calls.reverse()
        for name, call in calls:
            elapsed, peak, extra = timed(call, device)
            if sequence > warmup:
                samples[name].append(elapsed)
                memory[name].append((peak, extra))
    for states in (serial_states, batch_states):
        for state in states:
            assert state["last_input_seq"] == steps
            assert state["duplex_frame"].shape[0] == 1
            assert bool(torch.isfinite(state["duplex_frame"]).all())
        for key in CACHE_KEYS:
            assert len({state[key].untyped_storage().data_ptr() for state in states}) == batch
    serial, batched = distribution(samples["serial"]), distribution(samples["batched"])
    return {
        "batch": batch,
        "serial": serial,
        "batched": batched,
        "speedup_mean": serial["mean_ms"] / batched["mean_ms"],
        "samples_ms": samples,
        "memory": {
            name: {
                "peak_allocated_bytes": max(peak for peak, _ in values),
                "peak_step_extra_bytes": max(extra for _, extra in values),
            }
            for name, values in memory.items()
        },
        "stream_state_checks_passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    if not 0 <= args.warmup < args.steps or min(args.batches) < 1 or args.repeats < 1:
        parser.error("Require positive batches/repeats and 0 <= warmup < steps")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, weights = load_perception(args.model, device, getattr(torch, args.dtype))
    reference, reference_hash = load_reference(args.reference_source)
    audio = load_audio(args.audio)
    rows = []
    for repeat in range(1, args.repeats + 1):
        for batch in args.batches:
            row = benchmark(model, reference, audio, device, batch, args.steps, args.warmup)
            row["repeat"] = repeat
            rows.append(row)
            print(
                json.dumps(
                    {
                        "dtype": args.dtype,
                        "repeat": repeat,
                        "batch": batch,
                        "serial_ms": row["serial"]["mean_ms"],
                        "batched_ms": row["batched"]["mean_ms"],
                        "speedup": row["speedup_mean"],
                    }
                ),
                flush=True,
            )
    source = Path(thinker_module.__file__).resolve()
    head = subprocess.check_output(["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "head": head,
        "dtype": args.dtype,
        "gpu": current_platform.get_device_name(device.index or 0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_omni_version": importlib.metadata.version("vllm-omni"),
        "checkpoint_identity": checkpoint_identity(args.model),
        "weights": weights,
        "implementation_source": str(source),
        "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference_sha256": reference_hash,
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_scope": "synchronized whole-batch wall time; PCM/mel/Conformer/projection/cache commit",
        "memory_scope": "both A/B state sets retained; peaks are not standalone service capacity",
        "batches": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
