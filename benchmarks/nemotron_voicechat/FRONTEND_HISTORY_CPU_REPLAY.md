# VoiceChat frontend codec-history CPU replay

Companion validation for [PR #8688](https://github.com/vllm-project/vllm-omni/pull/8688) and [RFC #8687](https://github.com/vllm-project/vllm-omni/issues/8687). This validation branch carries the CPU reproducer and result summary separately from the PR. The PR implementation remains three files at `a1c92987e4ae9c0beefd6e060369b7a03b41379e`.

## Method

Measured on 2026-10-09 using Intel Xeon Gold 5320, CPU affinity 48, one PyTorch intra-op and inter-op thread, Python 3.12, PyTorch 2.13.0+cu129 and vLLM 0.31.0. Base: `4c5541cfc17143f80bdb89bbb7a5840b08bb52c6`. Fix: `a1c92987e4ae9c0beefd6e060369b7a03b41379e`. The matrix was specified before measurement: 375, 750 and 1,500 frames with one request, and 1,500 frames with four independent requests, each with three fresh-process runs per arm. Arm order alternates by case and repetition. All 24 runs are reported below.

The replay supplies cumulative prefixes with one new row of 31 int64 codes per 80 ms frame. The actual worker output builder uses the default Talker receiver role and the model's asynchronous duplex output-key policy. Actual frontend `MultimodalOutputProcessor.process_outputs()` calls handle native DELTA outputs with a resumable segment stop per frame. This matches the Talker's stop when the received timeline is exhausted; states remain live between segments. The CPU fixture bypasses GPU model inference and unrelated worker prefix-cache updates. Codec values and PCM are synthetic. Timing excludes imports, 20-frame warmup, payload construction, worker routing, and preservation checks; it covers the complete Talker frontend processor call, including accumulation, consolidation and completion construction. Worker, PCM processor, and codec-bearing consolidation timings are retained separately in locally generated reports.

The primary timing uses `time.thread_time_ns()`; wall time uses `time.perf_counter_ns()`. The common frontend and duplex-plugin sources are identical between arms. Every supplied inter-stage prefix and each of 51,750 PCM chunks is verified after processing. Each PCM chunk contains 1,764 samples at 22.05 kHz. This is a synthetic CPU replay, not a live inference or network-latency benchmark. Duration labels denote frame counts. Four request states exercise frontend scaling; the native deployment's default admission limit remains one session. Model GPU computation is unchanged.

## Results

The 120-second single-request median frontend thread CPU time is **97.915 s before and 27.25 ms after (3593×)**. Its last-quarter mean handler wall time is **151.45 ms before and 0.021 ms after**, compared with an 80 ms frame period under the measured CPU budget. At 1,500 frames, duplicated frontend codec retention is **34,898,250 elements / 266.3 MiB before and zero after**. Codec concatenation output volume is **130.2 GiB per request before and zero after**. The required complete inter-stage prefix retains all **46,500 elements**. The volume counter counts concatenation output tensor bytes, not RSS, network traffic, or memory-controller traffic. Empty consolidation guards still execute after suppression and are included in whole-processor timing.

| Frame timeline | Requests | Repetition | Arm | Frontend thread CPU, s | Frontend wall, s |
| --- | --- | --- | --- | --- | --- |
| 30 s | 1 | 1 | base | 1.290498 | 1.292176 |
| 30 s | 1 | 1 | fix | 0.006346 | 0.006685 |
| 30 s | 1 | 2 | fix | 0.006499 | 0.029543 |
| 30 s | 1 | 2 | base | 1.285701 | 1.301436 |
| 30 s | 1 | 3 | base | 1.309284 | 1.313073 |
| 30 s | 1 | 3 | fix | 0.006434 | 0.006880 |
| 60 s | 1 | 1 | fix | 0.012924 | 0.013763 |
| 60 s | 1 | 1 | base | 12.117772 | 12.127778 |
| 60 s | 1 | 2 | base | 11.950732 | 11.954217 |
| 60 s | 1 | 2 | fix | 0.013029 | 0.013753 |
| 60 s | 1 | 3 | fix | 0.013017 | 0.014098 |
| 60 s | 1 | 3 | base | 12.124624 | 12.126525 |
| 120 s | 1 | 1 | base | 97.947776 | 97.953923 |
| 120 s | 1 | 1 | fix | 0.027568 | 0.029202 |
| 120 s | 1 | 2 | fix | 0.027248 | 0.028932 |
| 120 s | 1 | 2 | base | 97.597220 | 97.605659 |
| 120 s | 1 | 3 | base | 97.914628 | 97.922338 |
| 120 s | 1 | 3 | fix | 0.027083 | 0.028754 |
| 120 s | 4 | 1 | fix | 0.063034 | 0.065014 |
| 120 s | 4 | 1 | base | 395.148350 | 395.621995 |
| 120 s | 4 | 2 | base | 392.733578 | 392.756730 |
| 120 s | 4 | 2 | fix | 0.061406 | 0.063453 |
| 120 s | 4 | 3 | fix | 0.061485 | 0.063478 |
| 120 s | 4 | 3 | base | 402.248269 | 402.425979 |

The rows retain all repetitions, including wall-time interruptions on the shared host. Summaries in the RFC and PR use three-run medians rather than a favorable individual pairing. Earlier RTServe snapshot-replacement timings are not the source of this table. The real-weight native duplex smoke in the PR is a separate functional check of live PCM delivery and response completion.

## Reproduction

Use a Linux environment with vLLM 0.31.0, matching vLLM-Omni dependencies, and pytest installed. No model weights or visible GPU are needed. Create separate checkouts at the base and fix SHAs. Keep the reproducer outside those checkouts and select the intended checkout through both the working directory and `PYTHONPATH`. The base checkout's existing worker-test helper is sufficient; applying the PR's test diff is unnecessary for this benchmark.

In the base checkout, use an available CPU core (the measurement used 48):

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" python /path/to/frontend_history_cpu_replay.py --arm base --revision 4c5541cfc17143f80bdb89bbb7a5840b08bb52c6 --steps 1500 --sessions 1 --cpu 48 --output /path/to/new-base-report.json
```

Run the corresponding command in the fix checkout:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" python /path/to/frontend_history_cpu_replay.py --arm fix --revision a1c92987e4ae9c0beefd6e060369b7a03b41379e --steps 1500 --sessions 1 --cpu 48 --output /path/to/new-fix-report.json
```

Repeat each pair in fresh processes three times, alternating arm order. To reproduce the full matrix, also use `--steps 375 --sessions 1`, `--steps 750 --sessions 1`, and `--steps 1500 --sessions 4`. Use a new output filename for each run; exclusive creation preserves existing reports. Sum `timing_ms.talker_frontend_cpu_ms` for thread CPU totals and `timing_ms.talker_frontend_wall_ms` for wall totals, dividing by 1,000 to obtain seconds. Take the median across the three totals in each arm. Compare the first and final quarters of each per-frame wall-time array for handler degradation.

The reproducer records the checked Git revision, software versions, affinity, source hashes, script hash, per-frame timings, retained codec size, copy volume, PCM counts and preservation assertions. The publication copy adds a license header and formatting; its parsed Python AST matches the measured script. Original report JSON remains outside both the PR and this validation branch.
