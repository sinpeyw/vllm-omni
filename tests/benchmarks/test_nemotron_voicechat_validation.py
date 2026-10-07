# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks for checkpoint provenance and native probe result contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pybase64
import pytest

from benchmarks.nemotron_voicechat.checkpoint_metadata import checkpoint_identity
from benchmarks.nemotron_voicechat.native_probe import summarize_audio
from vllm_omni.config.stage_config import resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_checkpoint_provenance_with_and_without_download_metadata(tmp_path: Path) -> None:
    config = b'{"model": {}}\n'
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "model.safetensors").write_bytes(b"unit-test placeholder")
    expected: dict[str, Any] = {
        "files": {
            "config.json": {"size_bytes": len(config), "sha256": hashlib.sha256(config).hexdigest()},
            "model.safetensors": {"size_bytes": 21},
        },
        "weight_content_hashed": False,
    }
    assert checkpoint_identity(tmp_path) == expected

    metadata = tmp_path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    for name in expected["files"]:
        (metadata / f"{name}.metadata").write_text("revision\nfile-etag\n123.0\n")
        expected["files"][name].update(hub_revision="revision", hub_etag="file-etag")
    assert checkpoint_identity(tmp_path) == expected

    (metadata / "model.safetensors.metadata").write_text("revision\n\n")
    with pytest.raises(ValueError, match="Incomplete Hub download metadata"):
        checkpoint_identity(tmp_path)


def audio_events() -> list[dict[str, Any]]:
    delta = pybase64.b64encode(np.full(1764, 8192, dtype="<i2").tobytes()).decode()
    return [
        {
            "type": "response.output_audio.delta",
            "delta": delta,
            "sample_rate_hz": 22050,
            "_client_received_at_s": arrival,
        }
        for arrival in [10.0, 10.1, 10.3, 10.4]
    ]


def test_native_probe_metrics_allow_extra_autonomous_packets() -> None:
    # Output-count completion accepts extra packets; it does not acknowledge
    # each input. Warmup discards output packets independently of input times.
    summary, pcm = summarize_audio(audio_events(), [9.0, 9.08, 9.16], warmup=1, finished_at=10.5)
    assert summary["input_frames"] == 3
    assert summary["audio_packets"] == 4
    assert len(pcm) == 4 * 3528
    assert summary["elapsed_s"] == pytest.approx(1.42)
    assert summary["drain_after_last_input_s"] == pytest.approx(1.34)
    assert summary["gap_p50_ms"] == pytest.approx(150)
    assert summary["gap_p95_ms"] == pytest.approx(195)
    assert summary["audio_rms"] == pytest.approx(0.25)


@pytest.mark.parametrize("invalid", ["count", "size", "rate", "base64"])
def test_native_probe_rejects_invalid_audio(invalid: str) -> None:
    events = audio_events()
    if invalid == "count":
        events = events[:2]
    elif invalid == "size":
        events[0]["delta"] = pybase64.b64encode(b"\x00\x00").decode()
    elif invalid == "rate":
        events[0]["sample_rate_hz"] = 16000
    else:
        events[0]["delta"] = "not valid base64!"
    with pytest.raises((AssertionError, ValueError)):
        summarize_audio(events, [9.0, 9.08, 9.16], warmup=1, finished_at=10.5)


def test_validation_profile_preserves_inherited_runtime_contracts() -> None:
    base = resolve_deploy_yaml(ROOT / "vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml")
    actual = resolve_deploy_yaml(ROOT / "benchmarks/nemotron_voicechat/configs/eager_four_sessions.yaml")
    # Compare the complete resolved profile, not just overridden fields.
    base["active_stream_window"] = 4
    base["duplex_session"]["max_sessions"] = 4
    for stage in base["stages"]:
        stage.update(max_num_seqs=4, enforce_eager=True)
        if stage["stage_id"] < 2:
            stage["max_model_len"] = 4096
    assert actual == base


@pytest.mark.parametrize("relative_l2, exit_code", [(0.0, 0), (0.03, 3)])
def test_component_cli_reports_completed_numerical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_l2: float, exit_code: int
) -> None:
    """Synthetic statistics test the CLI contract, not GPU numerical accuracy."""
    from benchmarks.nemotron_voicechat import validate_perception_batch as benchmark

    output = tmp_path / "result.json"
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"placeholder")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"placeholder")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            str(tmp_path),
            "--audio",
            str(audio),
            "--reference-source",
            str(tmp_path / "reference.py"),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "bfloat16",
            "--batches",
            "1",
            "--repeats",
            "1",
        ],
    )
    monkeypatch.setattr(benchmark, "load_perception", lambda *args: (None, {}))
    monkeypatch.setattr(benchmark, "load_reference", lambda *args: (None, "reference-sha256"))
    monkeypatch.setattr(benchmark, "load_audio", lambda *args: np.zeros(1280))
    monkeypatch.setattr(benchmark, "lifecycle_checks", lambda *args: {"errors": {}})
    monkeypatch.setattr(
        benchmark,
        "benchmark",
        lambda *args: {
            "errors": {"duplex_frame": {"max_relative_l2": relative_l2}},
        },
    )
    monkeypatch.setattr(benchmark.current_platform, "get_device_name", lambda *args: "CPU contract test")
    if exit_code:
        with pytest.raises(SystemExit) as error:
            benchmark.main()
        assert error.value.code == exit_code
    else:
        benchmark.main()
    assert json.loads(output.read_text())["passed"] is (exit_code == 0)

    # Reusing the result filename is a CLI error, before model loading.
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_native_observer_preserves_calls_and_reports_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Load the observer explicitly without installing its import finder in
    # this pytest process. Its wrappers must preserve arguments and returns.
    monkeypatch.delenv("VOICECHAT_PERCEPTION_TRACE", raising=False)
    source = ROOT / "benchmarks/nemotron_voicechat/_native_observer/sitecustomize.py"
    spec = importlib.util.spec_from_file_location("voicechat_validation_observer", source)
    assert spec is not None and spec.loader is not None
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setattr(observer, "DESTINATION", str(trace))
    calls = []
    output = object()

    def step(**kwargs: Any) -> object:
        calls.append(kwargs)
        return output

    @dataclass
    class Encoder:
        cache_aware_stream_step: Callable[..., object]

    @dataclass
    class Perception:
        encoder: Encoder

    class Thinker:
        def __init__(self, marker: str) -> None:
            self.marker = marker
            self.perception = Perception(Encoder(step))
            self._sessions: dict[str, dict[str, Any]] = {"a": {}, "b": {}}

        def on_requests_finished(self, ids: list[str]) -> int:
            for request_id in ids:
                self._sessions.pop(request_id)
            return len(ids)

    module = ModuleType("voicechat_observer_fixture")
    module.__file__ = str(source)
    monkeypatch.setattr(module, "NemotronVoiceChatThinkerForConditionalGeneration", Thinker, raising=False)
    observer.observe(module)
    model = Thinker("unchanged")
    signal = np.zeros((2, 80, 9), dtype=np.float32)
    assert model.marker == "unchanged"
    assert model.perception.encoder.cache_aware_stream_step(processed_signal=signal, drop_extra_pre_encoded=1) is output
    assert calls[0]["processed_signal"] is signal
    assert model.on_requests_finished(["a"]) == 1
    assert model.on_requests_finished(["b"]) == 1
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert records[1]["batch"] == 2
    assert records[1]["drop"] == 1
    assert records[-2]["remaining"] == ["b"]
    assert records[-1]["remaining"] == []
