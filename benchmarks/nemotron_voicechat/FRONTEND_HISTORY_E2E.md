# VoiceChat frontend history: real-weight end-to-end validation

Companion validation for [PR #8688](https://github.com/vllm-project/vllm-omni/pull/8688) and [RFC #8687](https://github.com/vllm-project/vllm-omni/issues/8687). This record is on a separate validation branch. No service launcher, raw result JSON, or benchmark artifact is added to the PR.

## Method

Measured on 2026-10-09 with one NVIDIA A800-SXM4-80GB, vLLM 0.31.0, and the unmodified `vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml`. All three native stages share the same GPU. Both arms use CPU affinity 26–51 and `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`. Base: `4c5541cfc17143f80bdb89bbb7a5840b08bb52c6`. Fix: `a1c92987e4ae9c0beefd6e060369b7a03b41379e`. The stock configuration, model computations, native automatic continuation policy, and client configuration are unchanged. Weights and tokenizer are reused from local storage with Hub access disabled.

Input is channel 0 of the checkpoint's `turn_taking.wav`, resampled to 16 kHz by the upstream fixture's `_read_wav()`. Each input contains 1,280 float32 samples (80 ms); the last source frame is zero-padded and the recording's frames repeat cyclically to supply 1,500 real input frames (120 seconds). WAV SHA256: `9602d5f78799644964b631e632c5740c325d518847cbb1ffb8917dee7abd17c1`. The six real-input streams have identical payload SHA256: `f1345cffde425b8e7f86cfb2a54f57951164c70c06da631014b03c21b700189c`. Inputs use absolute-clock 80 ms pacing. Client pacing-error P95 is 1.21–1.67 ms across the six measured runs.

Each arm starts a fresh server, passes a 50-frame pairing pilot, and runs a separate 200-frame warmup before three fresh 120-second sessions. Startup, pilots, warmup and teardown are excluded. The prespecified order is all three base sessions followed by all three fixed sessions, on the same GPU. Both servers exit cleanly after measurement. Two earlier setup attempts are excluded: one exposed invalid ordinal pairing, and the other was stopped when unrelated two-GPU experiments restarted. No result from either attempt enters the comparison. The final matrix starts after the other experiments finish and all GPUs remain idle for four minutes; the launcher monitors the two-GPU test cap during execution.

Native serving can insert silence input frames between real client inputs. Identical read-only observers in both arms record append `seq`, the existing `silence_continuation` flag, payload hashes and ordered PCM projection. The model's frame-aligned timeline maps an append sequence to its corresponding 80 ms codec output. PCM SHA256 verifies that each projected packet equals the packet received by the client. Every real input is matched through this timeline; automatic silence inputs are recorded separately. Native continuation counts can vary with timing and are reported below. The observers add the same recording path to both arms while leaving function arguments and returns, scheduling policy, caches and wire messages unchanged. All 9,000 real input frames receive matched PCM, with no server error events. Output packets contain 1,764 PCM16 samples at 22.05 kHz and are non-silent over each measured session.

## Metrics

**Client end-to-end audio completion RTF** = time from the first real input send to receipt of the PCM corresponding to the last real input, divided by the corresponding decoded PCM duration (120 seconds). It includes the entire inference, frontend and localhost WebSocket delivery path. Outputs corresponding to automatic silence inputs are excluded from the denominator. Real-time input pacing puts this completion metric near 1 when there is little backlog; it measures completion under a live input stream rather than peak offline generation throughput.

**Final-quarter input-to-PCM P95** = the nearest-rank 95th percentile of client PCM arrival minus its corresponding real input's send time, for the final 375 of 1,500 real frames. The last quarter covers the final 30 seconds of supplied input. It measures the delivery lag that a near-1 whole-session RTF can hide. It ends at client PCM receipt and excludes browser playback and semantic response-completion timing. Each summary is the median of the three run-level values.

## Results

| Client metric | Base median | Fix median |
| --- | --- | --- |
| Final-quarter input-to-PCM P95 | **1,777.55 ms** | **141.64 ms** |
| End-to-end audio completion RTF | **1.0151** | **1.0000** |

Final-quarter P95 is **12.55× lower (92.03% reduction)**. Whole-session RTF improves by **1.49%**; the dominant user-facing gain is lower delivery lag near the end of the session. The result does not imply a 12.55× increase in model generation throughput. Unchanged GPU computations are paired with suppression of unused client codec snapshots, which removes quadratic frontend accumulation and copying. The [isolated CPU replay](FRONTEND_HISTORY_CPU_REPLAY.md) documents that mechanism separately.

| Arm | Run | RTF | Final-quarter P95, ms | Completion, s | Planned automatic silence inputs | Received PCM packets |
| --- | --- | --- | --- | --- | --- | --- |
| base | 1 | 1.014309 | 1685.234 | 121.717 | 390 | 1886 |
| base | 2 | 1.015106 | 1777.549 | 121.813 | 410 | 1906 |
| base | 3 | 1.040464 | 5691.544 | 124.856 | 478 | 1972 |
| fix | 1 | 1.000486 | 141.277 | 120.058 | 359 | 1858 |
| fix | 2 | 1.000010 | 142.231 | 120.001 | 326 | 1825 |
| fix | 3 | 1.000028 | 141.636 | 120.003 | 338 | 1837 |

Automatic inputs include any continuation planned just before closing; PCM counts can include trailing automatic output. Metrics use only the 1,500 matched real-input frames. The baseline's per-run P95 varies from 1.69 to 5.69 seconds, while fixed P95 remains between 141 and 143 ms. All runs are retained; no favorable single pairing replaces the three-run median.

## Reproduction and provenance

Start each checkout's stock CLI with the same local weights, tokenizer reference and deployment YAML, exposing `/v1/realtime` on localhost. Use the upstream `DuplexClient` and `create_duplex_session_config()` defaults, including native automatic response handling. Supply the 1,500 frames at 80 ms intervals, record sends and audio-delta receipt times on the client's monotonic clock, and wait for every corresponding real-input PCM before closing. Do not stop at the first `response.done`, and do not pair input and output packet ordinals without distinguishing automatically inserted inputs. Repeat three fresh sessions per arm after warmup, on the same GPU and CPU settings.

The locally archived harness and read-only observer are outside both the implementation PR and this validation branch. Original result JSON retains all input timestamps, PCM timestamps, timeline sequences, payload/PCM hashes, observation rows, source revisions and functional assertions. `verified-results.json` independently recalculates both metrics from these records. Local artifact directory: `results/e2e-rtf-final-20261009-i74vwz8d` under the task workspace. Harness SHA256: `6449cdd5add8ab9e2b572c762347dadf5e23ce36fb66512560ac422c2c1b6b64`. Observer SHA256: `ccdc1e8ce56ce892e92e13530420ba9cb00385dc4bd70c9887d340288db1307d`.

| Raw record | SHA256 |
| --- | --- |
| `base/run-1.json` | `342b1279d7e8f7b15abf08c3df2af5697f6aff9ad0eafe127c73afea8f6791fb` |
| `base/run-2.json` | `3cbf0d4e8f7792691238f8f9648516f9c04c7db95b642e76bd8afc86ffbe9b0c` |
| `base/run-3.json` | `24dd6887257f3d11b50fcc6ecefb4e4e3bdf989a583d028ef78fb819dc574a04` |
| `fix/run-1.json` | `94142898d815812a946a11b47b198274d97e66979efe89ed34cba27928e14723` |
| `fix/run-2.json` | `b8496d973bb7d61dd085e9bae8dadf3e164e260cb1b16c290ecbab83d8ad87ba` |
| `fix/run-3.json` | `08d10f23062c3c4ae13caa6c4d16ed247349cb2328efb62c68a0dac706180e9e` |
