# VoiceChat GPU validation: 2026-10-08

Validation of [PR #8597](https://github.com/vllm-project/vllm-omni/pull/8597) at `acdb9c1e03e785cf6d2e2220d174d0a620d27c28`, against the unmodified singleton method from its upstream base `e54e185593ff84bb532fe8ca166147e142ae1f43`. The contribution remains three files; this evidence and tooling are on a separate validation branch.

## Environment and method

A800-SXM4-80GB, one GPU per experiment; Python 3.12.3, PyTorch 2.13.0+cu129 (CUDA 12.9), vLLM/vLLM-Omni 0.31.0, Transformers 5.14.1. All weights and the real `turn_taking.wav` fixture came from the local NVIDIA NemotronLabs VoiceChat 11B checkpoint, with the local Nemotron Nano 9B v2 tokenizer. No checkpoint was downloaded. Component loading strictly loaded 640 perception tensors. Final component timing used GPU 6; native eager used GPU 6 and native Graph used GPU 7. These final experiments used at most two GPUs concurrently.

The component comparison measures synchronized whole-batch wall time, including PCM preparation, streaming mel, FastConformer, adapter/projection, and cache commit. Each batch has distinct audio offsets per stream. Two 96-frame repetitions discard the first 16 frames each; singleton/batched order alternates every frame, yielding 160 measured samples per path. The initial component runs during server initialization remain in the local record; the results below use the publishable CLI after eager serving shut down.

## Component timings

| Dtype | Batch | Serial mean ± SD, ms | Batched mean ± SD, ms | Speedup |
| --- | --- | --- | --- | --- |
| float32 | 1 | 28.632 ± 0.258 | 28.777 ± 0.209 | 0.99x |
| float32 | 2 | 57.220 ± 0.426 | 30.159 ± 0.418 | 1.90x |
| float32 | 4 | 114.785 ± 1.584 | 30.625 ± 0.403 | 3.75x |
| float32 | 8 | 229.486 ± 1.688 | 31.532 ± 0.316 | 7.28x |
| bfloat16 | 1 | 28.968 ± 0.637 | 29.084 ± 0.484 | 1.00x |
| bfloat16 | 2 | 57.693 ± 0.535 | 30.841 ± 0.421 | 1.87x |
| bfloat16 | 4 | 115.491 ± 0.858 | 31.215 ± 0.364 | 3.70x |
| bfloat16 | 8 | 231.192 ± 1.461 | 31.808 ± 0.361 | 7.27x |

At batch 1 the measured whole-batch overhead is 0.5% in FP32 and 0.4% in BF16. Each row finished at the expected input sequence with a finite frame and independent per-stream cache storage.

### Memory

Both A/B state sets remain allocated while measuring. Peaks include the perception model and both sets; these are not standalone model-service capacity limits. Extra memory is peak allocation minus allocation immediately before a timed call.

| Dtype | Batch | Serial peak, MiB | Batched peak, MiB | Serial extra, MiB | Batched extra, MiB |
| --- | --- | --- | --- | --- | --- |
| float32 | 1 | 2420.7 | 2429.0 | 14.0 | 23.2 |
| float32 | 2 | 2435.7 | 2466.9 | 14.4 | 46.4 |
| float32 | 4 | 2465.5 | 2540.4 | 14.0 | 89.3 |
| float32 | 8 | 2525.8 | 2688.8 | 14.4 | 178.5 |
| bfloat16 | 1 | 1213.4 | 1217.4 | 7.1 | 11.1 |
| bfloat16 | 2 | 1220.7 | 1236.6 | 7.1 | 23.0 |
| bfloat16 | 4 | 1236.4 | 1275.2 | 7.1 | 46.4 |
| bfloat16 | 8 | 1267.1 | 1348.6 | 7.9 | 89.7 |

## Native full-duplex service

Each execution mode uses four clean native servers in flag-off/on/on/off order, with two paired rounds on the same GPU. Each server first runs 1/2/4-session calibration cohorts and a reopened session (17 warmup + 16 measured input frames per stream), then the same cohorts with 17 warmup + 128 measured frames. The 64 measured streams and 64 calibration streams all passed PCM packet, sample-rate, protocol, reopening, and final-empty-session-map checks. All eight servers shut down with exit code 0. The observed flag-off perception batches were always 1; flag-on measurement cohorts included batches 2 and 4.

The temporary profile raises admission, the active-stream window, and each stage's `max_num_seqs` from 1 to 4, and reduces the thinker/talker context limits to 4096. Memory fractions, native talker configuration, its 64-token budget, connectors, BF16 thinker, and FP32 talker remain inherited. The eager profile sets all stages eager; the Graph profile enables Graphs for thinker and talker, with successful capture confirmed in every server log. Acoustic perception remains eager. The upstream automatic-silence clock is unmodified; no RTServe or clock-control hook is loaded.

Each native table cell is mean completion time with the range of its two cohort means. Completion starts at the first post-warmup input and uses an output-audio-count threshold after input delivery. Autonomous silence may contribute output packets, so the times are delivery metrics rather than real-input acknowledgement or an 80 ms deadline result.

| Mode | Sessions | Flag off, mean [round range], s | Flag on, mean [round range], s |
| --- | --- | --- | --- |
| eager | 1 | 10.160 [10.160–10.161] | 10.161 [10.161–10.161] |
| eager | 2 | 13.283 [13.257–13.308] | 13.076 [12.905–13.247] |
| eager | 4 | 23.586 [22.736–24.437] | 18.608 [18.602–18.615] |
| graph | 1 | 10.160 [10.160–10.161] | 10.161 [10.160–10.162] |
| graph | 2 | 10.684 [10.570–10.798] | 10.163 [10.163–10.163] |
| graph | 4 | 21.709 [21.599–21.819] | 13.696 [13.632–13.760] |

### Output packet gaps

| Mode | Sessions | Flag-off per-stream P95 range, ms | Flag-on per-stream P95 range, ms |
| --- | --- | --- | --- |
| eager | 1 | 74.3–80.1 | 67.7–72.4 |
| eager | 2 | 104.1–131.4 | 112.3–114.0 |
| eager | 4 | 194.7–238.6 | 143.9–166.4 |
| graph | 1 | 88.8–89.8 | 90.7–93.1 |
| graph | 2 | 89.8–107.3 | 75.4–97.4 |
| graph | 4 | 164.5–199.6 | 113.3–134.0 |

N4 completion improved in both paired rounds (mean reductions 21.1% eager, 36.9% Graph), while N1 completion remained at the input-pacing floor. Packet-gap improvements are not uniform: the second eager N2 round is about 10 ms worse despite slightly shorter completion, and Graph N1 gaps are a few milliseconds higher. The count-based completion and the unmodified autonomous clock do not isolate a causal latency effect; this result supports the component optimization and delivery/lifecycle correctness, not uniform tail-latency or deadline guarantees. The model option remains disabled by default.

## CPU and code checks

At `acdb9c1e03e785cf6d2e2220d174d0a620d27c28`, the complete VoiceChat/runner CPU command below passed: **99 passed, 1 skipped, 10 deselected, 14 warnings in 6.36 s**. The skip requires optional audio libraries. All applicable local pre-commit gates passed on the three contribution files. Source fingerprints and clean worktree status were verified after GPU, CPU, and lint execution.

Upstream DCO and documentation checks passed. The upstream pre-commit job did not start because GitHub failed to acquire a runner five times; its empty step list and annotations confirm an infrastructure failure. This is not reported as an upstream lint pass. A rerun request returned that the run could not be rerun.

## Reproduction

Use a matching existing environment, local checkpoint/tokenizer, a free A800 80 GB GPU, and a fresh output directory. Check out the PR at the pinned head in `VOICECHAT_PR_REPO`; check out this validation branch in `VOICECHAT_VALIDATION_REPO`. Export only its benchmark package into a tools directory so the archived production model cannot shadow the PR model.

```bash
export VOICECHAT_PR_REPO=/path/to/pr-checkout
export VOICECHAT_VALIDATION_REPO=/path/to/validation-checkout
export VOICECHAT_PYTHON=/path/to/matching/.venv/bin/python
export VOICECHAT_MODEL=/path/to/NVIDIA-NemotronLabs-VoiceChat-11B
export NEMOTRON_VOICECHAT_LLM_PATH=/path/to/NVIDIA-Nemotron-Nano-9B-v2-tokenizer
export VOICECHAT_TOOLS=/path/to/fresh-tools
export VOICECHAT_RUN=/path/to/fresh-results
mkdir "$VOICECHAT_TOOLS" "$VOICECHAT_RUN"
git -C "$VOICECHAT_VALIDATION_REPO" archive HEAD benchmarks | tar -x -C "$VOICECHAT_TOOLS"
export PYTHONPATH="$VOICECHAT_TOOLS:$VOICECHAT_PR_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 VLLM_NO_USAGE_STATS=1
export VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=240
git -C "$VOICECHAT_PR_REPO" show e54e185593ff84bb532fe8ca166147e142ae1f43:vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_thinker.py > "$VOICECHAT_RUN/baseline_thinker.py"
CUDA_VISIBLE_DEVICES=0 "$VOICECHAT_PYTHON" "$VOICECHAT_TOOLS/benchmarks/nemotron_voicechat/time_perception_batch.py" \
  --model "$VOICECHAT_MODEL" --audio "$VOICECHAT_MODEL/turn_taking.wav" \
  --reference-source "$VOICECHAT_RUN/baseline_thinker.py" \
  --dtype float32 --batches 1 2 4 8 --steps 96 --warmup 16 --repeats 2 \
  --output "$VOICECHAT_RUN/component-float32.json"
```

Repeat the component command with `--dtype bfloat16` and a new output filename. Select an actually available GPU instead of assuming device 0 is free.

For native serving, create both profiles from the checked-in validation profile with the PR deployment as their base:

```bash
"$VOICECHAT_PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml
tools = Path(os.environ["VOICECHAT_TOOLS"])
repo = Path(os.environ["VOICECHAT_PR_REPO"])
run = Path(os.environ["VOICECHAT_RUN"])
for mode in ("eager", "graph"):
    config = yaml.safe_load((tools / "benchmarks/nemotron_voicechat/configs/eager_four_sessions.yaml").read_text())
    config["base_config"] = str(repo / "vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml")
    for stage in config["stages"]:
        stage["enforce_eager"] = mode == "eager" or stage["stage_id"] == 2
    (run / f"{mode}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

Run a fresh server for each flag setting. Choose a fresh case directory and free port; the recorded order is `false`, `true`, `true`, `false` for each profile. In a server terminal:

```bash
export VOICECHAT_GPU=0 VOICECHAT_PORT=18375 VOICECHAT_MODE=eager VOICECHAT_BATCHED=false
export VOICECHAT_CASE="$VOICECHAT_RUN/eager-r1-off"
mkdir "$VOICECHAT_CASE"
cd "$VOICECHAT_PR_REPO"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  CUDA_VISIBLE_DEVICES="$VOICECHAT_GPU" NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 \
  VOICECHAT_PERCEPTION_TRACE="$VOICECHAT_CASE/perception.jsonl" \
  PYTHONPATH="$VOICECHAT_TOOLS/benchmarks/nemotron_voicechat/_native_observer:$PYTHONPATH" \
  "$VOICECHAT_PYTHON" -m vllm_omni.entrypoints.cli.main serve "$VOICECHAT_MODEL" --omni \
  --host 127.0.0.1 --port "$VOICECHAT_PORT" --served-model-name voicechat-pr-validation \
  --deploy-config "$VOICECHAT_RUN/$VOICECHAT_MODE.yaml" \
  --stage-overrides "{\"0\":{\"hf_overrides\":{\"batch_duplex_perception\":$VOICECHAT_BATCHED}}}"
```

After `/health` is ready, use the same case/port variables in a client terminal:

```bash
NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 \
  "$VOICECHAT_PYTHON" "$VOICECHAT_TOOLS/benchmarks/nemotron_voicechat/native_probe.py" \
  --url "ws://127.0.0.1:$VOICECHAT_PORT/v1/realtime" --model-dir "$VOICECHAT_MODEL" \
  --audio "$VOICECHAT_MODEL/turn_taking.wav" --output "$VOICECHAT_CASE/calibration" \
  --perception-trace "$VOICECHAT_CASE/perception.jsonl" --batches 1 2 4 --frames 16 --warmup 17
NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 \
  "$VOICECHAT_PYTHON" "$VOICECHAT_TOOLS/benchmarks/nemotron_voicechat/native_probe.py" \
  --url "ws://127.0.0.1:$VOICECHAT_PORT/v1/realtime" --model-dir "$VOICECHAT_MODEL" \
  --audio "$VOICECHAT_MODEL/turn_taking.wav" --output "$VOICECHAT_CASE/measured" \
  --perception-trace "$VOICECHAT_CASE/perception.jsonl" --batches 1 2 4 --frames 128 --warmup 17
```

Add `--expect-batched` for flag-on probes. Close only the server you started, then repeat with a fresh case directory and distinct free port. Graph capture applies to the native language/talker stages, not the perception helper. The observer records batch sizes and cleanup without GPU synchronization or scheduling changes.

From the PR root, the CPU/lint checks are:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD" "$VOICECHAT_PYTHON" -m pytest -q \
  tests/model_executor/models/test_nemotron_voicechat_perception_batch.py \
  tests/model_executor/models/test_nemotron_voicechat_perception_window.py \
  tests/model_executor/models/test_nemotron_voicechat_shapes.py \
  tests/model_executor/models/test_nemotron_voicechat_registration.py \
  tests/model_executor/models/test_nemotron_voicechat_talker_replay.py \
  tests/model_executor/models/nemotron_voicechat/duplex \
  tests/worker/test_omni_gpu_model_runner.py \
  -m 'core_model and cpu' --run-level=core_model

pre-commit run --show-diff-on-failure --files \
  vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_thinker.py \
  tests/model_executor/models/test_nemotron_voicechat_perception_batch.py \
  recipes/NVIDIA/NemotronLabs-VoiceChat.md
```

AI assistance: OpenAI Codex assisted with benchmark tooling, validation, and this record.
