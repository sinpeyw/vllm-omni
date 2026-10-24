"""Compare real-weight, client-observed delivery at exact base and PR commits."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import median
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent
SETUP = json.loads((ROOT / "setup.json").read_text())
PERIOD_NS = 80_000_000
CPU_AFFINITY = set(range(26, 52))


def save(path: Path, data: object) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def p95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


class LedgerReader:
    def __init__(self, path: Path, session_id: str) -> None:
        self.stream = path.open()
        self.session_id = session_id
        self.pending = ""
        self.rows: list[dict] = []

    def refresh(self) -> list[dict]:
        self.pending += self.stream.read()
        lines = self.pending.split("\n")
        self.pending = lines.pop()
        for line in lines:
            row = json.loads(line)
            if row.get("session_id") == self.session_id:
                self.rows.append(row)
        return self.rows


def pair_frames(
    rows: list[dict], input_hashes: list[str], pcm_hashes: list[str]
) -> list[int | None]:
    appends = [row for row in rows if row["kind"] == "append"]
    assert all(isinstance(row["automatic_silence"], bool) for row in appends), (
        "Missing append context"
    )
    assert [row["model_seq"] for row in appends] == list(range(1, len(appends) + 1)), (
        "Non-contiguous append sequence"
    )
    assert len({row["epoch"] for row in appends}) <= 1, "Unexpected session reset"
    real = [row for row in appends if not row["automatic_silence"]]
    assert len(real) <= len(input_hashes), "Unexpected external append"
    assert [row["payload_sha256"] for row in real] == input_hashes[: len(real)], (
        "Input payload mismatch"
    )
    pcm = [row for row in rows if row["kind"] == "pcm"]
    assert [row["model_seq"] for row in pcm] == list(range(1, len(pcm) + 1)), (
        "Non-contiguous PCM frame sequence"
    )
    visible = min(len(pcm), len(pcm_hashes))
    assert [row["pcm_sha256"] for row in pcm[:visible]] == pcm_hashes[:visible], (
        "Projected PCM differs from wire PCM"
    )
    assert all(
        row["pcm_bytes"] == 3528 and row["audio_duration_ms"] == 80 for row in pcm
    )
    assert all(row["sample_rate_hz"] == 22050 for row in pcm)
    indices = {row["model_seq"]: index for index, row in enumerate(pcm[:visible])}
    return [indices.get(row["model_seq"]) for row in real]


async def measure(args: argparse.Namespace) -> None:
    import numpy as np

    import vllm_omni
    from tests.e2e.online_serving.nemotron_voicechat_realtime_duplex import (
        DEFAULT_INSTRUCTIONS,
        _read_wav,
    )
    from vllm_omni.clients.duplex import DuplexClient
    from vllm_omni.clients.nemotron_voicechat import create_duplex_session_config

    assert Path(vllm_omni.__file__).resolve().parent.parent == Path(args.checkout)
    pcm = _read_wav(Path(SETUP["model"]) / "turn_taking.wav", input_channel=0)
    frames = [
        np.pad(
            pcm[start : start + 1280],
            (0, max(0, 1280 - pcm[start : start + 1280].size)),
        )
        .astype("<f4")
        .tobytes()
        for start in range(0, pcm.size, 1280)
    ]
    sent_ns: list[int] = []
    received_ns: list[int] = []
    input_hashes: list[str] = []
    pcm_hashes: list[str] = []
    input_digest = hashlib.sha256()
    audio_digest = hashlib.sha256()
    output_samples: list[int] = []
    errors: list[dict] = []
    events: Counter[str] = Counter()
    statuses: list[str | None] = []
    wire_sequences: list[int] = []
    rms_sum = 0.0

    class MeasuredClient(DuplexClient):
        async def _dispatch(self, data):
            nonlocal rms_sum
            received = time.monotonic_ns()
            kind = data.get("type", "")
            events[kind] += 1
            sequence = data.get("server_event_seq")
            if isinstance(sequence, int):
                wire_sequences.append(sequence)
            if kind == "error":
                errors.append(data)
            elif kind == "response.done":
                statuses.append(data.get("response", {}).get("status"))
            elif kind == "response.output_audio.delta":
                raw = base64.b64decode(data["delta"], validate=True)
                assert data["sample_rate_hz"] == 22050
                assert data["format"] == "pcm16" and len(raw) == 3528
                received_ns.append(received)
                output_samples.append(len(raw) // 2)
                pcm_hashes.append(hashlib.sha256(raw).hexdigest())
                audio_digest.update(raw)
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
                rms_sum += float(np.dot(samples, samples))
            await super()._dispatch(data)

    client = MeasuredClient(
        f"ws://127.0.0.1:{args.port}/v1/realtime",
        model="voicechat-e2e-history",
        config=create_duplex_session_config(
            instructions=DEFAULT_INSTRUCTIONS, idle_timeout_s=900
        ),
        handshake_timeout_s=120,
        reconnect=None,
        heartbeat_interval_s=None,
    )
    result = {
        "passed": False,
        "arm": args.arm,
        "frames": args.frames,
        "frame_period_ms": 80,
        "checkout": args.checkout,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True
        ).strip(),
        "metric_definitions": SETUP["planned_metrics"],
        "errors": errors,
    }
    failure: BaseException | None = None
    reader = None
    try:
        async with client:
            assert client.session_info["capabilities"]["chunk_period_ms"] == 80
            result["session_id"] = client.session_id
            reader = LedgerReader(Path(args.ledger), client.session_id)
            anchor_ns = time.monotonic_ns() + 100_000_000
            for tick in range(args.frames):
                await asyncio.sleep(
                    max(0, (anchor_ns + tick * PERIOD_NS - time.monotonic_ns()) / 1e9)
                )
                if errors:
                    raise RuntimeError(errors)
                frame = frames[tick % len(frames)]
                input_digest.update(frame)
                input_hashes.append(hashlib.sha256(frame).hexdigest())
                sent_ns.append(time.monotonic_ns())
                await client.append_audio(frame)
                if (tick + 1) % 250 == 0:
                    print(
                        f"{args.arm}: real input {tick + 1}/{args.frames}, received PCM {len(received_ns)}",
                        flush=True,
                    )
            deadline = time.monotonic() + 600
            previous_progress = -1
            while True:
                if errors:
                    raise RuntimeError(errors)
                pairs = pair_frames(reader.refresh(), input_hashes, pcm_hashes)
                paired_count = sum(index is not None for index in pairs)
                if len(pairs) == args.frames and paired_count == args.frames:
                    break
                if paired_count // 250 != previous_progress:
                    previous_progress = paired_count // 250
                    print(
                        f"{args.arm}: draining, paired real PCM {paired_count}/{args.frames}, total PCM {len(received_ns)}",
                        flush=True,
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Only {paired_count}/{args.frames} corresponding PCM frames received"
                    )
                await asyncio.sleep(0.05)

            assert len(sent_ns) == len(pairs) == args.frames and all(
                index is not None for index in pairs
            )
            assert wire_sequences == sorted(set(wire_sequences)) and not errors
            matched_ns = [received_ns[index] for index in pairs]
            matched_samples = [output_samples[index] for index in pairs]
            latency_ms = [
                (arrival - sent) / 1e6
                for sent, arrival in zip(sent_ns, matched_ns, strict=True)
            ]
            assert min(latency_ms) >= 0
            last_quarter = latency_ms[-math.ceil(args.frames / 4) :]
            duration_s = sum(matched_samples) / 22050
            delivery_s = (matched_ns[-1] - sent_ns[0]) / 1e9
            appends = [row for row in reader.rows if row["kind"] == "append"]
            real = [row for row in appends if not row["automatic_silence"]]
            result.update(
                passed=True,
                client_e2e_rtf=delivery_s / duration_s,
                final_quarter_frame_latency_p95_ms=p95(last_quarter),
                client_total_delivery_s=delivery_s,
                final_input_to_last_pcm_s=(matched_ns[-1] - sent_ns[-1]) / 1e9,
                corresponding_audio_duration_s=duration_s,
                paired_real_pcm_frames=len(pairs),
                all_frame_latency_p95_ms=p95(latency_ms),
                first_audio_latency_ms=latency_ms[0],
                final_frame_latency_ms=latency_ms[-1],
                pcm_rms=math.sqrt(rms_sum / sum(output_samples)),
                audio_sha256=audio_digest.hexdigest(),
                input_payload_sha256=input_digest.hexdigest(),
                pacing_error_p95_ms=p95(
                    [
                        (sent - anchor_ns - tick * PERIOD_NS) / 1e6
                        for tick, sent in enumerate(sent_ns)
                    ]
                ),
                input_to_pcm_latency_ms=latency_ms,
                paired_pcm_packet_indices=pairs,
                real_input_model_sequences=[row["model_seq"] for row in real],
                automatic_silence_planned=sum(
                    row["automatic_silence"] for row in appends
                ),
                thinker_projected_frames=sum(
                    row["kind"] == "thinker" for row in reader.rows
                ),
                frame_pairing="Read-only native source_input_seq ledger maps each real input to its ordered codec frame; exact PCM hashes verify projection-to-client identity. Automatic silence is tracked separately.",
            )
    except BaseException as error:
        failure = error
        result["passed"] = False
        result["failure"] = repr(error)
    finally:
        result.update(
            input_frames_sent=len(sent_ns),
            output_pcm_frames=len(received_ns),
            event_counts=dict(events),
            response_statuses=statuses,
            input_send_ns=sent_ns,
            pcm_received_ns=received_ns,
            input_packet_sha256=input_hashes,
            pcm_packet_sha256=pcm_hashes,
            output_samples_per_packet=output_samples,
            observation_rows=reader.refresh() if reader else [],
        )
        save(Path(args.output), result)
        if reader:
            reader.stream.close()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in [
                    "arm",
                    "passed",
                    "input_frames_sent",
                    "output_pcm_frames",
                    "paired_real_pcm_frames",
                    "client_e2e_rtf",
                    "final_quarter_frame_latency_p95_ms",
                    "failure",
                ]
            }
        ),
        flush=True,
    )
    if failure:
        raise failure


def server_environment(checkout: Path, arm: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            env.pop(key)
    interpreter = Path(SETUP["interpreter"])
    temp = Path(SETUP["short_tmp"]) / arm
    temp.mkdir(exist_ok=False)
    cache = Path(SETUP["compiled_cache"])
    env.update(
        VIRTUAL_ENV=str(interpreter.parent.parent),
        PATH=str(interpreter.parent) + os.pathsep + env["PATH"],
        PYTHONPATH=str(checkout),
        CUDA_VISIBLE_DEVICES="7",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
        NEMOTRON_VOICECHAT_LLM_PATH=SETUP["tokenizer"],
        VLLM_PLUGINS="vllm_omni_register_models",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        NO_PROXY="localhost,127.0.0.1,::1",
        no_proxy="localhost,127.0.0.1,::1",
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        XDG_CACHE_HOME=str(cache),
        TMPDIR=str(temp),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        VOICECHAT_OBSERVATION_LEDGER=str(ROOT / arm / "observations.jsonl"),
        FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
        TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
        NUMBA_CACHE_DIR=str(cache / "numba"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def run_arm(arm: str, checkout: Path, port: int) -> list[dict]:
    folder = ROOT / arm
    folder.mkdir(exist_ok=False)
    env = server_environment(checkout, arm)
    config = checkout / "vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml"
    command = [
        SETUP["interpreter"],
        str(ROOT / "observed_server.py"),
        "serve",
        SETUP["model"],
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "voicechat-e2e-history",
        "--deploy-config",
        str(config),
    ]
    save(
        folder / "server-provenance.json",
        {
            "command": command,
            "environment": {
                key: env[key]
                for key in [
                    "VIRTUAL_ENV",
                    "PYTHONPATH",
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "HF_HUB_OFFLINE",
                    "TMPDIR",
                    "VOICECHAT_OBSERVATION_LEDGER",
                ]
            },
            "cpu_affinity": sorted(CPU_AFFINITY),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "observer_sha256": hashlib.sha256(
                (ROOT / "observed_server.py").read_bytes()
            ).hexdigest(),
        },
    )
    opener = build_opener(ProxyHandler({}))
    results = []
    with (folder / "server.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=checkout,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: os.sched_setaffinity(0, CPU_AFFINITY),
        )
        with (folder / "server.pid").open("x") as stream:
            stream.write(str(process.pid) + "\n")
        print(f"Starting {arm} on GPU 7, server PID {process.pid}", flush=True)
        started = time.monotonic()
        try:
            ready = False
            while time.monotonic() - started < 1000:
                check_gpu_cap()
                if process.poll() is not None:
                    raise RuntimeError(f"{arm} server exited with {process.returncode}")
                try:
                    with opener.open(
                        f"http://127.0.0.1:{port}/health", timeout=2
                    ) as response:
                        ready = response.status == 200
                except (OSError, TimeoutError):
                    pass
                if ready:
                    break
                time.sleep(2)
            if not ready:
                raise TimeoutError(f"{arm} server never became healthy")
            print(f"{arm} ready after {time.monotonic() - started:.1f}s", flush=True)
            for label, frames in [("pilot", 50), ("warmup", 200)] + [
                (f"run-{repeat}", 1500) for repeat in range(1, 4)
            ]:
                output = folder / f"{label}.json"
                client_command = [
                    SETUP["interpreter"],
                    str(Path(__file__).resolve()),
                    "client",
                    "--arm",
                    arm,
                    "--checkout",
                    str(checkout),
                    "--port",
                    str(port),
                    "--frames",
                    str(frames),
                    "--ledger",
                    env["VOICECHAT_OBSERVATION_LEDGER"],
                    "--output",
                    str(output),
                ]
                measurement = subprocess.Popen(
                    client_command,
                    cwd=checkout,
                    env=env,
                    preexec_fn=lambda: os.sched_setaffinity(0, CPU_AFFINITY),
                )
                measurement_started = time.monotonic()
                try:
                    while measurement.poll() is None:
                        check_gpu_cap()
                        if time.monotonic() - measurement_started > 850:
                            raise TimeoutError(f"{arm} {label} measurement timed out")
                        time.sleep(1)
                    if measurement.returncode:
                        raise subprocess.CalledProcessError(
                            measurement.returncode, client_command
                        )
                finally:
                    if measurement.poll() is None:
                        measurement.send_signal(signal.SIGINT)
                        try:
                            measurement.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            measurement.kill()
                            measurement.wait(timeout=10)
                data = json.loads(output.read_text())
                assert (
                    data["passed"]
                    and data["input_frames_sent"]
                    == data["paired_real_pcm_frames"]
                    == frames
                )
                assert data["corresponding_audio_duration_s"] == frames * 0.08
                if label.startswith("run-"):
                    results.append(data)
                time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            save(
                folder / "retired.json",
                {
                    "server_pid": process.pid,
                    "returncode": process.returncode,
                    "elapsed_s": time.monotonic() - started,
                },
            )
            print(f"{arm} server stopped, exit {process.returncode}", flush=True)
    return results


def gpu_usage() -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        int(row.split(",")[0]): int(row.split(",")[1]) for row in output.splitlines()
    }


def check_gpu_cap() -> None:
    others = [gpu for gpu, memory in gpu_usage().items() if gpu != 7 and memory >= 100]
    if len(others) >= 2:
        raise RuntimeError(
            f"Other GPUs {others} became active; stop the owned service to maintain the two-GPU test cap"
        )


def other_scheduler_experiments() -> list[int]:
    parents = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            arguments = (folder / "cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeDecodeError):
            continue
        if any(argument.endswith("/scheduler_ablation.py") for argument in arguments):
            parents.append(int(folder.name))
    return parents


def matrix() -> None:
    started = time.monotonic()
    quiet_since = None
    while True:
        usage = gpu_usage()
        assert usage[7] < 100, "GPU 7 is occupied; preserve existing processes"
        busy = [gpu for gpu, memory in usage.items() if memory >= 100]
        other_parents = other_scheduler_experiments()
        if busy or other_parents:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        quiet_s = 0 if quiet_since is None else time.monotonic() - quiet_since
        if quiet_s >= SETUP["gpu_quiet_period_s"]:
            break
        print(
            f"Waiting for stable test availability: GPUs {busy}, other experiment parents {other_parents}, quiet {quiet_s:.0f}/{SETUP['gpu_quiet_period_s']}s",
            flush=True,
        )
        if time.monotonic() - started > 3600:
            raise TimeoutError(
                "GPU experiments still active; no third test GPU was started"
            )
        time.sleep(30)
    results = {}
    for arm, checkout in [
        ("base", Path(SETUP["base_checkout"])),
        ("fix", Path(SETUP["fix_checkout"])),
    ]:
        assert not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=checkout, text=True
        )
        assert len([memory for memory in gpu_usage().values() if memory >= 100]) < 2
        results[arm] = run_arm(arm, checkout, SETUP["port"])
    summary = {
        arm: {
            metric: median(run[metric] for run in runs)
            for metric in [
                "client_e2e_rtf",
                "final_quarter_frame_latency_p95_ms",
                "client_total_delivery_s",
            ]
        }
        for arm, runs in results.items()
    }
    assert (
        len({run["input_payload_sha256"] for runs in results.values() for run in runs})
        == 1
    )
    save(
        ROOT / "summary.json",
        {
            "setup": SETUP,
            "summary": summary,
            "raw_files": {
                arm: [f"{arm}/run-{index}.json" for index in range(1, 4)]
                for arm in results
            },
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["client", "matrix"])
    parser.add_argument("--arm")
    parser.add_argument("--checkout")
    parser.add_argument("--port", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--ledger")
    parser.add_argument("--output")
    options = parser.parse_args()
    if options.mode == "client":
        asyncio.run(measure(options))
    else:
        matrix()
