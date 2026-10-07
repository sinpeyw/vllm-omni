# VoiceChat duplex perception validation

These tools validate the experimental `batch_duplex_perception` model override.
It defaults to `false`. They use local weights and the native vLLM-Omni path;
they do not download checkpoints or provide a new scheduler.

Run commands from the repository root in an environment matching this checkout
(the recorded runs used vLLM 0.31.0). Set local checkpoint paths first:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 VLLM_NO_USAGE_STATS=1 VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=240
export VOICECHAT_MODEL=/path/to/NVIDIA-NemotronLabs-VoiceChat-11B
export NEMOTRON_VOICECHAT_LLM_PATH=/path/to/NVIDIA-Nemotron-Nano-9B-v2-tokenizer
```

The checkpoint must contain `config.json`, `model.safetensors`, and
`turn_taking.wav`. The component benchmark loads only perception tensors.
Native serving needs the full checkpoint, tokenizer, one available GPU
(the recorded profile used an A800 80 GB), and a free port.

## Component comparison

Supply a separate checkout of the singleton reference. Recorded measurements
use upstream commit `d5a3380103df2fc827f095c60dbfb6e1c5655fd3`.
Choose a fresh output filename for every run:

```bash
export VOICECHAT_BASE=/path/to/reference-checkout
PYTHONPATH="$PWD" python benchmarks/nemotron_voicechat/validate_perception_batch.py \
  --model "$VOICECHAT_MODEL" --audio "$VOICECHAT_MODEL/turn_taking.wav" \
  --reference-source "$VOICECHAT_BASE/vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_thinker.py" \
  --dtype float32 --batches 1 2 4 8 --steps 96 --warmup 16 --repeats 2 \
  --output results/perception-fp32.json
```

Repeat with `--dtype bfloat16` and another output filename. Each batch uses
different waveform offsets per row. Timing synchronizes around the whole
batch: PCM preparation, mel extraction, Conformer, projection, and cache commit.
Both A/B state sets remain allocated during memory measurements; reported
peaks are not isolated engine or whole-service VRAM limits.

JSON records source/input hashes, configuration SHA256, file sizes, and any
local Hugging Face download revision/ETag. Download metadata identifies the
recorded source; the large weight file is not freshly content-hashed. Copied
checkpoints without that metadata still report configuration SHA256 and sizes.

| Exit code | Meaning |
| --- | --- |
| 0 | Numerical comparison and lifecycle checks passed. |
| 1 | Unhandled runtime failure, such as loading or an assertion; inspect the traceback. |
| 2 | Invalid CLI arguments, including an existing output path. |
| 3 | Comparison completed and JSON was written, but the relative-L2 gate failed. |

The current gates are `1e-4` for FP32 and `0.02` for BF16. They check numerical
agreement, not generated speech quality. An assertion during lifecycle checks
is a runtime failure, distinct from a completed comparison above its tolerance.

## Native service comparison

[eager_four_sessions.yaml](configs/eager_four_sessions.yaml) inherits the shipped
duplex profile. It raises admission, active-stream window, and each stage's
`max_num_seqs` from 1 to 4, reduces the two AR context limits to 4096, and
uses eager execution. Stage placement, BF16 thinker, FP32 talker, connectors,
and memory fractions remain inherited. This is an experimental validation
profile; shipped defaults and capability advertisement stay unchanged.

Use a dedicated test server. In terminal 1, select an available GPU and a fresh
result directory, then start flag-off serving:

```bash
export VOICECHAT_GPU=0
export VOICECHAT_RUN="$PWD/results/native-serial"
mkdir -p "$PWD/results"
mkdir "$VOICECHAT_RUN"
CUDA_VISIBLE_DEVICES="$VOICECHAT_GPU" \
  VOICECHAT_PERCEPTION_TRACE="$VOICECHAT_RUN/perception.jsonl" \
  PYTHONPATH="$PWD/benchmarks/nemotron_voicechat/_native_observer:$PWD" \
  NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 \
  vllm-omni serve "$VOICECHAT_MODEL" --omni \
  --host 127.0.0.1 --port 18365 --served-model-name voicechat-pr-validation \
  --deploy-config benchmarks/nemotron_voicechat/configs/eager_four_sessions.yaml \
  --stage-overrides '{"0":{"hf_overrides":{"batch_duplex_perception":false}}}'
```

If inherited proxy settings prevent local startup, remove those variables
only for the test command with `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
-u http_proxy -u https_proxy -u all_proxy`; preserve the user's shell settings.

After `/health` becomes ready, run terminal 2 from the same repository root
with the same model path and `VOICECHAT_RUN` value:

```bash
PYTHONPATH="$PWD" NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 \
  python benchmarks/nemotron_voicechat/native_probe.py \
  --url ws://127.0.0.1:18365/v1/realtime \
  --model-dir "$VOICECHAT_MODEL" --audio "$VOICECHAT_MODEL/turn_taking.wav" \
  --output "$VOICECHAT_RUN/client" \
  --perception-trace "$VOICECHAT_RUN/perception.jsonl" \
  --batches 1 2 4 --frames 128 --warmup 17
```

Stop that server with Ctrl-C. Repeat with a fresh `VOICECHAT_RUN`, set the
stage override to `true`, and add `--expect-batched` to the probe. This compares
flag off/on on the same patched checkout. The component comparison uses the
separate pristine singleton reference.

The opt-in observer records actual encoder batch sizes and request cleanup
without GPU synchronization. It is installed only in the dedicated server's
`PYTHONPATH`; it is not part of normal serving. The probe checks valid 22.05 kHz
PCM16 packets, delivery for 1/2/4 clients, reopening, and an empty final model
session map. It preserves per-stream audio/events and a summary JSON.

Elapsed time starts at the first post-warmup input send and ends at an output
audio-count threshold. Drain starts at the last input send. Packet gaps discard
the first 17 **output** packets. The autonomous silence clock can emit extra
packets, so these metrics do not establish per-input acknowledgement, speech
quality, or an 80 ms realtime deadline. A delivery pass is not a performance pass.

## Recorded evidence

[reference_results.json](reference_results.json) preserves summaries from
implementation commit `72063750f2f4bbfdcc618ecfdf99b757418182ac`, including
per-repetition component distributions and per-stream native metrics. These
historical measurements are separate from checks of later tooling changes.

FP32 B2/B4/B8 component speedups were 1.90x / 3.74x / 7.26x. BF16 stable-frame
drift reached 4.88% / 5.49% / 7.43%, failing the 2% gate. One native eager A/B
round improved N4 but regressed N2 (elapsed about 14.3 s to 15.3–15.4 s;
packet-gap P95 about 109 ms to 148 ms). The N2 cause and BF16 speech-quality
impact remain unresolved. Graph-enabled multi-session, ASR/WER, and sustained
capacity studies have not been established by this evidence.
