# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Probe concurrent native VoiceChat delivery and lifecycle on an existing server.

Uses the public duplex client and the upstream fixture reader. Audio-count
completion measures delivery mechanics, not real-input acknowledgement,
speech quality, or an 80 ms realtime deadline.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pybase64

from benchmarks.nemotron_voicechat.checkpoint_metadata import checkpoint_identity
from tests.e2e.online_serving.nemotron_voicechat_realtime_duplex import _read_wav
from vllm_omni.clients.duplex import DuplexClient, EventCollector, wait_for_condition, write_pcm16_wav
from vllm_omni.clients.nemotron_voicechat import create_duplex_session_config


def summarize_audio(
    events: list[dict[str, Any]], input_times: list[float], warmup: int, finished_at: float
) -> tuple[dict[str, Any], bytes]:
    """Validate packet delivery and report elapsed/drain and inter-packet gaps."""
    packets = [event for event in events if event.get("type") == "response.output_audio.delta"]
    audio = [pybase64.b64decode(event["delta"], validate=True) for event in packets]
    if len(input_times) - warmup < 2:
        raise ValueError("At least two measured input frames are required")
    if len(packets) < len(input_times) or {len(packet) for packet in audio} != {3528}:
        raise AssertionError(f"Expected at least {len(input_times)} valid 3,528-byte output packets")
    if {event.get("sample_rate_hz") for event in packets} != {22050}:
        raise AssertionError("Expected 22.05 kHz output audio")
    gaps = np.diff([float(event["_client_received_at_s"]) for event in packets[warmup:]]) * 1000
    pcm = b"".join(audio)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    return {
        "input_frames": len(input_times),
        "audio_packets": len(packets),
        "elapsed_s": finished_at - input_times[warmup],
        "drain_after_last_input_s": finished_at - input_times[-1],
        "audio_rms": float(np.sqrt(np.mean(samples**2))),
        "gap_p50_ms": float(np.median(gaps)),
        "gap_p95_ms": float(np.percentile(gaps, 95)),
    }, pcm


async def cohort(args: argparse.Namespace, audio: np.ndarray, count: int, name: str) -> list[dict[str, Any]]:
    """Open a cohort together, pace inputs, collect valid audio, and close it."""
    barrier = asyncio.Event()
    ready: asyncio.Queue[int] = asyncio.Queue()

    async def stream(row: int) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
        collector = EventCollector()
        config = create_duplex_session_config(
            instructions="You are NVIDIA VoiceChat. Answer briefly. Start by greeting the user.", idle_timeout_s=300
        )
        client = DuplexClient(args.url, model=args.model, config=config, handshake_timeout_s=120, reconnect=None)
        consumer = asyncio.create_task(collector.consume(client))
        await asyncio.sleep(0)
        try:
            async with client:
                await ready.put(row)
                await barrier.wait()
                start = time.monotonic()
                input_times = []
                total = args.warmup + args.frames
                for frame in range(total):
                    await asyncio.sleep(max(0, start + frame * 0.08 - time.monotonic()))
                    samples = np.take(audio, np.arange(1280) + (frame + row * 19) * 1280, mode="wrap")
                    input_times.append(time.monotonic())
                    await client.append_audio(samples.astype("<f4").tobytes())
                await client.commit()
                await wait_for_condition(
                    lambda: bool(collector.errors()) or collector.count("response.output_audio.delta") >= total,
                    timeout_s=90,
                    label=f"{name}/{row} audio delivery",
                )
                if collector.errors():
                    raise AssertionError(collector.errors())
                result, pcm = summarize_audio(collector.events, input_times, args.warmup, time.monotonic())
                result.update(
                    row=row,
                    session_id=client.session_id,
                    errors=collector.errors(),
                    event_counts=dict(Counter(str(event.get("type")) for event in collector.events)),
                    capabilities=client.session_info.get("capabilities"),
                    transcript="".join(
                        str(event.get("delta", ""))
                        for event in collector.events
                        if event.get("type") == "response.output_audio_transcript.delta"
                    ),
                )
                return result, pcm, collector.events.copy()
        finally:
            if not consumer.done():
                consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer

    tasks = [asyncio.create_task(stream(row)) for row in range(count)]
    try:
        for _ in tasks:
            await asyncio.wait_for(ready.get(), 120)
        barrier.set()
        streams = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Save only after every stream stops measuring; file I/O must not inflate
    # peers' packet gaps on the shared client event loop.
    for result, pcm, events in streams:
        folder = args.output / name / str(result["row"])
        folder.mkdir(parents=True, exist_ok=False)
        write_pcm16_wav(folder / "output.wav", pcm, sample_rate_hz=22050)
        (folder / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
        print(json.dumps({"cohort": name, "result": result}), flush=True)
    return [result for result, _, _ in streams]


async def run(args: argparse.Namespace) -> None:
    audio = _read_wav(args.audio, input_channel=0)
    identity = checkpoint_identity(args.model_dir) if args.model_dir is not None else None
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for count in args.batches:
        results.append({"sessions": count, "streams": await cohort(args, audio, count, f"n{count}")})
        await asyncio.sleep(2)
    results.append({"sessions": 1, "reopen": True, "streams": await cohort(args, audio, 1, "reopen")})
    await asyncio.sleep(2)
    summary = {
        "protocol_checks_passed": True,
        "checkpoint_identity": identity,
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "warmup": args.warmup,
        "frames": args.frames,
        "client": results,
        "completion": "output audio-count threshold; autonomous silence frames may be included",
    }
    if args.perception_trace is not None:
        trace = [json.loads(line) for line in args.perception_trace.read_text().splitlines()]
        sizes = Counter(record["batch"] for record in trace if record["kind"] == "perception")
        finished = [record for record in trace if record["kind"] == "finished"]
        if not finished or finished[-1]["remaining"]:
            raise AssertionError("Model sessions remain after closing all probe clients")
        if args.expect_batched and not any(size > 1 for size in sizes):
            raise AssertionError("No batched perception calls were observed")
        summary.update(batch_histogram=dict(sizes), finished_events=len(finished), last_finished_remaining=[])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="voicechat-pr-validation")
    parser.add_argument(
        "--model-dir", type=Path, help="Local checkpoint for provenance; no model is loaded by the client"
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perception-trace", type=Path, help="Observer trace from a dedicated local test server")
    parser.add_argument("--expect-batched", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=17)
    args = parser.parse_args()
    if args.frames < 2 or args.warmup < 0 or min(args.batches) < 1 or len(set(args.batches)) != len(args.batches):
        parser.error("Require frames >= 2, warmup >= 0, and distinct positive batch sizes")
    if args.output.exists():
        parser.error(f"Select a new output directory: {args.output}")
    if args.expect_batched and args.perception_trace is None:
        parser.error("--expect-batched requires --perception-trace")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
