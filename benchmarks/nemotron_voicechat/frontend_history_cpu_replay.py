# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: E402
# Pin the process before importing Torch and vLLM for consistent CPU placement.

"""CPU replay of upstream VoiceChat worker routing and segment output handling.

This external validation script is not part of PR #8688. It uses synthetic
integer codec prefixes and PCM, not model inference or a network service.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--sessions", type=int, required=True)
parser.add_argument("--arm", choices=("base", "fix"), required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--cpu", type=int, default=48)
args = parser.parse_args()
os.sched_setaffinity(0, {args.cpu})

import numpy as np
import pytest
import torch
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from tests.worker.test_gpu_ar_model_runner import _make_async_output_runner
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_talker import (
    NemotronVoiceChatTalkerForConditionalGeneration,
)
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
repo = Path.cwd()
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert revision == args.revision, (revision, args.revision)
production_files = [
    "vllm_omni/worker/gpu_ar_model_runner.py",
    "vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_talker.py",
    "vllm_omni/model_executor/models/nemotron_voicechat/duplex/plugin.py",
    "vllm_omni/outputs/output_processor.py",
    "vllm_omni/outputs/mm_outputs.py",
]
assert not subprocess.check_output(["git", "diff", "HEAD", "--", *production_files])


def make_processor(req_ids, modality):
    processor = MultimodalOutputProcessor(None, log_stats=False, engine_core_output_type=modality)
    for rid in req_ids:
        processor.request_states[rid] = OmniRequestState(
            request_id=rid,
            external_req_id=rid,
            parent_req=None,
            request_index=0,
            lora_request=None,
            prompt=None,
            prompt_token_ids=[0],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            output_kind=RequestOutputKind.DELTA,
        )
    return processor


def replay(steps, sessions, *, observe):
    req_ids = [f"r{i + 1}" for i in range(sessions)]
    talker_processor = make_processor(req_ids, "latent")
    audio_processor = make_processor(req_ids, "audio")
    runner = _make_async_output_runner(engine_output_type="latent")
    runner.model_config.stage_id = 1
    runner.model_config.session_mode = "duplex"
    runner.model_config.stage_connector_config = {"extra": {"role": "receiver"}}
    runner._pooler_payload_include_hidden_flag = False
    runner.requests = dict.fromkeys(req_ids, object())
    talker = object.__new__(NemotronVoiceChatTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.vllm_config = runner.vllm_config
    runner.model = talker
    policy = getattr(talker, "omni_client_multimodal_output_keys", None)
    assert policy == (() if args.arm == "fix" else None)
    generator = torch.Generator().manual_seed(8688)
    codes = [torch.randint(0, 1024, (steps, 31), generator=generator) for _ in req_ids]
    pcm = torch.arange(1764, dtype=torch.float32) / 1764
    schedule = SimpleNamespace(
        total_num_scheduled_tokens=sessions,
        num_scheduled_tokens=dict.fromkeys(req_ids, 1),
    )
    hidden = torch.zeros(sessions, 1)
    timing = {
        key: []
        for key in (
            "worker_wall_ms",
            "talker_frontend_wall_ms",
            "talker_frontend_cpu_ms",
            "pcm_frontend_wall_ms",
            "consolidation_wall_ms",
            "consolidation_cpu_ms",
        )
    }
    copied_bytes = 0
    original_consolidate = OmniRequestState._consolidate_multimodal_tensors

    def observe_consolidation(state):
        nonlocal copied_bytes
        if state.mm_type != "latent":
            return original_consolidate(state)
        value = state.mm_accumulated.tensors.get("codes.audio")
        if isinstance(value, list) and len(value) > 1:
            copied_bytes += sum(t.numel() * t.element_size() for t in value)
        wall_start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        original_consolidate(state)
        cpu_end = time.thread_time_ns()
        wall_end = time.perf_counter_ns()
        timing["consolidation_wall_ms"].append((wall_end - wall_start) / 1e6)
        timing["consolidation_cpu_ms"].append((cpu_end - cpu_start) / 1e6)

    with pytest.MonkeyPatch.context() as patch:
        # The CPU fixture bypasses GPU inference and unrelated prefix-cache updates.
        # Routing, payload partitioning, accumulation, consolidation, DELTA drain,
        # completion construction, and segment lifecycle execute real source code.
        patch.setattr(GPUARModelRunner, "_resolve_pooler_payload_req_ids", lambda self, ids: ("latent", ids))
        patch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
        patch.setattr(GPUARModelRunner, "_process_additional_information_updates", lambda *a, **kw: None)
        if observe:
            patch.setattr(OmniRequestState, "_consolidate_multimodal_tensors", observe_consolidation)
        for step in range(1, steps + 1):
            prefixes = [history[:step].clone() for history in codes]
            payload = {
                "codes": {"audio": prefixes},
                "meta": {
                    "nvc_logical_prompt_len": [torch.tensor([37]) for _ in req_ids],
                    "codec_streaming": [torch.tensor([True]) for _ in req_ids],
                },
            }
            start = time.perf_counter_ns()
            output = runner._build_omni_model_runner_output_from_snapshot(
                scheduler_output=schedule,
                hidden_states=hidden,
                staged_hidden_states_cpu=None,
                multimodal_outputs=payload,
                req_ids_output_copy=req_ids,
                req_id_to_index_output_copy={rid: i for i, rid in enumerate(req_ids)},
                valid_sampled_token_ids=[[1] for _ in req_ids],
                logprobs_lists=None,
                prompt_logprobs_dict={},
                num_nans_in_logits=None,
                kv_connector_output=None,
                ec_connector_output=None,
                cudagraph_stats=None,
                kv_extracted_req_ids=None,
                num_scheduled_tokens_np=np.ones(sessions, dtype=np.int32),
                query_start_loc_cpu=torch.arange(sessions, dtype=torch.long),
            )
            timing["worker_wall_ms"].append((time.perf_counter_ns() - start) / 1e6)
            assert len(output.inter_stage_outputs) == sessions
            for i, prefix in enumerate(prefixes):
                assert torch.equal(output.inter_stage_outputs[i]["codes.audio"], prefix)
            engine_outputs = [
                OmniEngineCoreOutput(
                    request_id=rid,
                    new_token_ids=[1],
                    multimodal_output=output.multimodal_outputs[i] if output.multimodal_outputs else None,
                    finish_reason=FinishReason.STOP,
                    is_segment_finished=True,
                )
                for i, rid in enumerate(req_ids)
            ]
            audio_outputs = [
                OmniEngineCoreOutput(
                    request_id=rid,
                    new_token_ids=[1],
                    multimodal_output={
                        "model_outputs": pcm,
                        "sr": torch.tensor(22050),
                        "meta.tts_is_last_chunk": torch.tensor([0]),
                    },
                    finish_reason=FinishReason.STOP,
                    is_segment_finished=True,
                )
                for rid in req_ids
            ]
            wall_start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            processed = talker_processor.process_outputs(engine_outputs)
            cpu_end = time.thread_time_ns()
            wall_end = time.perf_counter_ns()
            timing["talker_frontend_wall_ms"].append((wall_end - wall_start) / 1e6)
            timing["talker_frontend_cpu_ms"].append((cpu_end - cpu_start) / 1e6)
            start = time.perf_counter_ns()
            delivered = audio_processor.process_outputs(audio_outputs)
            timing["pcm_frontend_wall_ms"].append((time.perf_counter_ns() - start) / 1e6)
            assert len(processed.request_outputs) == len(delivered.request_outputs) == sessions
            assert not processed.reqs_to_abort and not delivered.reqs_to_abort
            for response in delivered.request_outputs:
                assert torch.equal(response.outputs[0].multimodal_output["audio"], pcm)
            assert len(talker_processor.request_states) == len(audio_processor.request_states) == sessions
    elements = [
        state.mm_accumulated.tensors.get("codes.audio", torch.empty(0)).numel()
        for state in talker_processor.request_states.values()
    ]
    expected = 31 * steps * (steps + 1) // 2 if args.arm == "base" else 0
    assert elements == [expected] * sessions, (elements, expected)
    expected_copied = sessions * 31 * 8 * (steps * (steps + 1) * (steps + 2) // 6 - 1) if args.arm == "base" else 0
    if observe:
        assert copied_bytes == expected_copied, (copied_bytes, expected_copied)
    return {
        "timing_ms": timing,
        "max_frontend_codec_elements_per_session": expected,
        "frontend_codec_bytes_per_session": expected * 8,
        "codec_concatenation_output_bytes": copied_bytes,
        "inter_stage_final_codec_elements_per_session": steps * 31,
        "pcm_chunks_delivered": steps * sessions,
        "pcm_samples_per_chunk": 1764,
        "pcm_sample_rate_hz": 22050,
        "all_pcm_and_inter_stage_prefixes_verified": True,
    }


replay(20, args.sessions, observe=False)
gc.collect()
run_start = time.perf_counter()
result = replay(args.steps, args.sessions, observe=True)
run_elapsed = time.perf_counter() - run_start
result.update(
    arm=args.arm,
    revision=revision,
    repo=str(repo),
    input_duration_s=args.steps * 0.08,
    steps=args.steps,
    sessions=args.sessions,
    complete_replay_wall_s=run_elapsed,
    python=sys.version,
    interpreter=sys.executable,
    torch=torch.__version__,
    cpu=platform.processor(),
    affinity=sorted(os.sched_getaffinity(0)),
    torch_threads=torch.get_num_threads(),
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    source_sha256={path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in production_files},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope=(
        "Synthetic CPU output-path replay at exact base/fix commits; "
        "no GPU inference, network timing, or end-to-end speedup claim."
    ),
)
with args.output.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(
    json.dumps(
        {
            "arm": args.arm,
            "steps": args.steps,
            "sessions": args.sessions,
            "frontend_wall_s": sum(result["timing_ms"]["talker_frontend_wall_ms"]) / 1000,
            "frontend_cpu_s": sum(result["timing_ms"]["talker_frontend_cpu_ms"]) / 1000,
            "frontend_codec_elements_per_session": result["max_frontend_codec_elements_per_session"],
            "pcm_chunks_delivered": result["pcm_chunks_delivered"],
        }
    ),
    flush=True,
)
