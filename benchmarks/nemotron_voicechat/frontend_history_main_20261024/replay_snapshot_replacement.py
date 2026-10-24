"""Control only the upstream snapshot-replacement call on actual CPU tensors.

This is a source-level accumulation replay, not another live-service run.
The payload implementation and replacement helper are the downloaded,
unchanged production files from main 9bc166c. Their output-modality and
tensor utility imports resolve through the original publication checkout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "latest/vllm_omni/outputs"
os.sched_setaffinity(0, {48})
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


payloads = load("audited_main_payloads", SOURCE / "mm_outputs.py")
policy = load("audited_main_snapshot_policy", SOURCE / "multimodal_accumulation.py")
Payload = payloads.MultimodalPayload
Modality = payloads.OutputModality


def replay(*, replacement: bool) -> dict:
    accumulated = Payload()
    history = torch.arange(1500 * 31).reshape(1500, 31)
    codec_concat_calls = 0
    codec_concat_output_bytes = 0
    started = time.perf_counter()
    for step in range(1, 1501):
        incoming = Payload.from_dict(
            {
                "codes.audio": history[:step].clone(),
                "meta.nvc_logical_prompt_len": torch.tensor([37]),
                "meta.nvc_tts_step": torch.tensor([step]),
                "meta.codec_streaming": torch.tensor([True]),
            }
        )
        policy.replace_snapshot_keys(accumulated, incoming)
        if replacement:
            policy.replace_nemotron_snapshot_keys(accumulated, incoming)
        accumulated = accumulated.merged_with(incoming)
        codes = accumulated["codes.audio"]
        if isinstance(codes, list) and len(codes) > 1:
            codec_concat_calls += 1
            codec_concat_output_bytes += sum(t.numel() * t.element_size() for t in codes)
        accumulated.consolidate_tensors(Modality.LATENT)
        accumulated.consolidate_metadata()
        if replacement:
            assert torch.equal(accumulated["codes.audio"], history[:step])
            assert accumulated["meta.nvc_tts_step"].tolist() == [step]
        if step % 250 == 0:
            print(replacement, step, accumulated["codes.audio"].numel(), flush=True)
    elements = accumulated["codes.audio"].numel()
    expected = 1500 * 31 if replacement else 31 * 1500 * 1501 // 2
    assert elements == expected
    assert codec_concat_calls == (0 if replacement else 1499)
    return {
        "replacement_enabled": replacement,
        "frames": 1500,
        "retained_codec_elements": elements,
        "retained_codec_bytes": elements * 8,
        "codec_concat_calls": codec_concat_calls,
        "codec_concat_output_bytes": codec_concat_output_bytes,
        "replay_wall_s": time.perf_counter() - started,
    }


result = {
    "scope": "Actual main payload merge/consolidation and snapshot helper; synthetic CPU prefixes; only replacement call toggled; no inference or service-latency claim.",
    "main": "9bc166cd556cdaa4e0b18543da0c2def0076275e",
    "introduced_by": "1c0bde507dfbffd25b32f6092dbe4526c945b7dd (#8758)",
    "torch": torch.__version__,
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "source_sha256": {
        name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        for name in ("mm_outputs.py", "multimodal_accumulation.py", "output_processor.py")
    },
    "current_main": replay(replacement=True),
    "same_code_without_replacement_call": replay(replacement=False),
}
with (ROOT / "snapshot-replacement-replay.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result, indent=2), flush=True)
