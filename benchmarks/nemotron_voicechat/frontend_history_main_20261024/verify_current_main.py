"""Recalculate the original input-to-PCM metric from every formal raw run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "validated-run"


def p95(values: list[float]) -> float:
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def verify_run(path: Path, revision: str) -> dict:
    data = json.loads(path.read_text())
    assert data["passed"] and not data["errors"]
    assert data["revision"] == revision
    assert data["frames"] == data["input_frames_sent"] == 1500
    assert data["paired_real_pcm_frames"] == 1500
    assert data["corresponding_audio_duration_s"] == 120
    assert data["pcm_rms"] > 0
    sent = data["input_send_ns"]
    arrivals = data["pcm_received_ns"]
    assert len(sent) == 1500 and all(b > a for a, b in zip(sent, sent[1:]))
    assert all(b >= a for a, b in zip(arrivals, arrivals[1:]))
    appends = [row for row in data["observation_rows"] if row["kind"] == "append"]
    pcm = [row for row in data["observation_rows"] if row["kind"] == "pcm"]
    assert [row["model_seq"] for row in appends] == list(range(1, len(appends) + 1))
    assert [row["model_seq"] for row in pcm] == list(range(1, len(pcm) + 1))
    assert all(isinstance(row["automatic_silence"], bool) for row in appends)
    assert len({row["epoch"] for row in appends}) == 1
    real = [row for row in appends if not row["automatic_silence"]]
    assert len(real) == 1500
    assert [row["payload_sha256"] for row in real] == data["input_packet_sha256"]
    visible_pcm = pcm[: len(arrivals)]
    assert len(visible_pcm) == len(arrivals)
    assert [row["pcm_sha256"] for row in visible_pcm] == data["pcm_packet_sha256"]
    assert all(
        row["pcm_bytes"] == 3528
        and row["sample_rate_hz"] == 22050
        and row["audio_duration_ms"] == 80
        for row in visible_pcm
    )
    by_sequence = {row["model_seq"]: index for index, row in enumerate(visible_pcm)}
    paired_indices = [by_sequence[row["model_seq"]] for row in real]
    assert paired_indices == data["paired_pcm_packet_indices"]
    latency = [
        (arrivals[index] - started) / 1e6
        for started, index in zip(sent, paired_indices, strict=True)
    ]
    assert min(latency) >= 0 and latency == data["input_to_pcm_latency_ms"]
    final = p95(latency[-375:])
    assert final == data["final_quarter_frame_latency_p95_ms"]
    return {
        "path": str(path.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": revision,
        "first_quarter_p95_ms": p95(latency[:375]),
        "final_quarter_p95_ms": final,
        "pacing_error_p95_ms": data["pacing_error_p95_ms"],
        "automatic_silence_planned": data["automatic_silence_planned"],
        "received_pcm_frames": len(arrivals),
        "paired_real_pcm_frames": len(paired_indices),
        "pcm_rms": data["pcm_rms"],
        "input_payload_sha256": data["input_payload_sha256"],
    }


def main() -> None:
    setup = json.loads((RUN / "setup.json").read_text())
    arms = {}
    for arm in ("base", "fix"):
        runs = [verify_run(RUN / arm / f"run-{n}.json", setup[arm]) for n in range(1, 4)]
        retired = json.loads((RUN / arm / "retired.json").read_text())
        assert retired["returncode"] in (0, -15)
        arms[arm] = {
            "first_quarter_p95_median_ms": median(r["first_quarter_p95_ms"] for r in runs),
            "final_quarter_p95_median_ms": median(r["final_quarter_p95_ms"] for r in runs),
            "final_quarter_p95_range_ms": [
                min(r["final_quarter_p95_ms"] for r in runs),
                max(r["final_quarter_p95_ms"] for r in runs),
            ],
            "runs": runs,
        }
    all_runs = [r for arm in arms.values() for r in arm["runs"]]
    assert len({r["input_payload_sha256"] for r in all_runs}) == 1
    assert sum(r["paired_real_pcm_frames"] for r in all_runs) == 9000
    result = {
        "base": setup["base"],
        "fix": setup["fix"],
        "arms": arms,
        "late_p95_ratio": arms["base"]["final_quarter_p95_median_ms"]
        / arms["fix"]["final_quarter_p95_median_ms"],
        "verified_real_frames": 9000,
    }
    with (ROOT / "verified-results.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
