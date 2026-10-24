# VoiceChat frontend history: current-main recheck

Companion evidence for [PR #8688](https://github.com/vllm-project/vllm-omni/pull/8688), [RFC #8687](https://github.com/vllm-project/vllm-omni/issues/8687), and the [request to rerun the original input-to-PCM harness on current main](https://github.com/vllm-project/vllm-omni/pull/8688#pullrequestreview-5632020195). This report and its artifacts are on a separate validation branch; none is added to the implementation PR.

## Result

The 2026-10-24 recheck does **not** reproduce the old multi-second late-session growth or establish an end-to-end speedup from #8688 on current main. The summary is the median of three run-level nearest-rank P95 values in each arm. Every run is retained.

| Client input-to-corresponding-PCM P95 | Current main | Current main plus #8688 |
| --- | --- | --- |
| First 375 real input frames | 87.91 ms | 88.73 ms |
| Last 375 real input frames | **87.40 ms** | **89.50 ms** |
| Range of the three last-window P95 values | 83.40–137.69 ms | 84.10–264.43 ms |

All 9,000 formal real input frames have matched PCM, all sessions deliver non-silent audio, and no error events occurred. The independently recalculated values and per-run record hashes are in [verified-results.json](frontend_history_main_20261024/verified-results.json). The current-main worker module passes **93 CPU tests**. The historical [vLLM 0.31 / October 9 measurement](FRONTEND_HISTORY_E2E.md) remains historical evidence; its 12.6× result is not a current-main claim.

## Why the old bottleneck is gone

[#8758](https://github.com/vllm-project/vllm-omni/pull/8758) merged on October 18 as [`1c0bde5`](https://github.com/vllm-project/vllm-omni/commit/1c0bde507dfbffd25b32f6092dbe4526c945b7dd). Its changes include the snapshot fix separately proposed in [#8772](https://github.com/vllm-project/vllm-omni/pull/8772), the isolated PR was closed when #8758 merged rather than merged separately. The relevant [processor call](https://github.com/vllm-project/vllm-omni/blob/9bc166cd556cdaa4e0b18543da0c2def0076275e/vllm_omni/outputs/output_processor.py#L170) invokes [replace_nemotron_snapshot_keys](https://github.com/vllm-project/vllm-omni/blob/9bc166cd556cdaa4e0b18543da0c2def0076275e/vllm_omni/outputs/multimodal_accumulation.py#L78) before merging each payload. A payload marked with `nvc_logical_prompt_len` replaces its prior `codes.audio` and per-wake metadata, instead of appending another complete prefix.

A targeted CPU replay uses the unchanged current-main payload merge/consolidation implementation and replacement helper with 1,500 synthetic `[frames, 31]` int64 prefixes. Only the replacement call is toggled. With replacement, the final accumulator contains **46,500 elements**, the current prefix. Without it, the accumulator contains **34,898,250 elements** and performs 1,499 codec concatenations. This verifies that the merged fix directly removes the quadratic accumulation mechanism; this comparison is **not** a performance benefit attributed to #8688 and is not a live-service ablation. The replay's payload tensor utilities match current main; the imported output-modality helper differs only in the unused `meta.finished` override, with identical latent/code concatenation rules for these inputs. [Replay source](frontend_history_main_20261024/replay_snapshot_replacement.py) and [result](frontend_history_main_20261024/snapshot-replacement-replay.json) are retained separately from the PR.

Other changes between the October 9 baseline and this main revision also affect runtime behavior. This analysis does not attribute every change in absolute latency to snapshot replacement. The directly established points are that the original duplication bug has an upstream fix and that current-main service measurements no longer demonstrate the original #8688 speedup.

#8688 still suppresses unused async-duplex Talker client payloads earlier in the worker. It preserves complete Code2Wav inter-stage histories and final PCM. This is now an output-routing cleanup rather than the fix responsible for removing current-main's quadratic accumulation.

## Same original end-to-end metric

Base source is main `9bc166cd556cdaa4e0b18543da0c2def0076275e`. Fixed source is that same main plus [the exact applied PR patch](frontend_history_main_20261024/validated-diff.patch); the experiment recorded local revision `5defa1a66dcac8d136a1de6252055f853e96292b`. [Base](frontend_history_main_20261024/base-source-provenance.json) and [fixed](frontend_history_main_20261024/fix-source-provenance.json) source manifests record the SHA256 of each changed file and the stock deployment YAML before launch. The public PR head remains `a1c92987e4ae9c0beefd6e060369b7a03b41379e`; no validation files or merge commit were pushed to it.

The run uses one NVIDIA A800-SXM4-80GB, vLLM 0.32.0, PyTorch 2.13.0+cu129, and the unmodified `vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml`. All native stages share physical GPU 7. CPU affinity is 26–51, with `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`. The independent environment uses vLLM's required FlashInfer 0.7.0.post1 core package without the optional precompiled cubin/cache packages; runtime compilation occurs before measurement. An earlier initialization-only attempt failed with incompatible inherited optional FlashInfer packages and produced no pilot or formal data.

Each arm starts a fresh server, passes a separate 50-frame pairing pilot and 200-frame warmup, then runs three fresh sessions of 1,500 real input frames. The sequence is all base sessions followed by all fixed sessions. Each frame is 1,280 float32 samples at 16 kHz, paced against an absolute monotonic clock every 80 ms. Channel 0 of the real checkpoint's `turn_taking.wav` is resampled with the upstream fixture, its last frame zero-padded, and its frames cycled to 120 seconds. WAV SHA256 is `9602d5f78799644964b631e632c5740c325d518847cbb1ffb8917dee7abd17c1`; the six real-input payload streams share SHA256 `f1345cffde425b8e7f86cfb2a54f57951164c70c06da631014b03c21b700189c`. Existing local weights/tokenizer are reused with Hub access disabled.

The metric runs from each real input's send timestamp to receipt of its corresponding 80 ms PCM packet, not the gap between output packets. The unchanged read-only observer uses native append sequences and the automatic-silence marker to identify real input, then PCM SHA256 to verify projected versus wire packet identity. All real inputs must match before a session closes. Nearest-rank P95 is calculated separately over the first and final 375 real inputs; browser playback and semantic response completion are outside this metric. Native automatic silence remains enabled and is excluded from real-input pairing.

The measurement functions retain the original harness AST, and the observer is byte-identical to the original published experiment. Adaptations are checkout/interpreter/asset paths, a free localhost port, task-owned cache locations and a 30-second pretrial quiet period. Startup, compilation, pilot and warmup do not enter the formal samples. Both servers exit cleanly with code 0, and all GPUs are idle after the matrix.

| Arm | Run | First-window P95, ms | Final-window P95, ms | Pacing-error P95, ms | Planned automatic inputs |
| --- | --- | --- | --- | --- | --- |
| base | 1 | 92.719 | 137.686 | 1.105 | 33 |
| base | 2 | 85.017 | 87.396 | 1.161 | 18 |
| base | 3 | 87.912 | 83.401 | 1.059 | 16 |
| fix | 1 | 95.060 | 89.500 | 1.081 | 20 |
| fix | 2 | 81.866 | 264.429 | 1.067 | 27 |
| fix | 3 | 88.732 | 84.096 | 1.066 | 10 |

## Reproduction and raw evidence

Check out main `9bc166c` for the base, and apply `validated-diff.patch` to a second checkout for the fixed arm. Keep the [harness and observer](frontend_history_main_20261024/validated-run) outside both source checkouts. Adapt the machine-specific source, interpreter, model, tokenizer, temporary/cache and port paths in `setup.json` and the launcher's physical GPU/CPU affinity as needed; preserve the workload, metric, pairing and warmup. Launch the matrix with the matching vLLM 0.32 environment. The recorded stock config SHA256 is `664f305c7667661aa3791613db64f48fcd6ba3fe4b4b359693ae2dd69d393bfa` in both arms.

The six original formal JSON records are archived as `.json.gz` under [base](frontend_history_main_20261024/validated-run/base) and [fix](frontend_history_main_20261024/validated-run/fix). Decompress them to their original `.json` paths before running [verify_current_main.py](frontend_history_main_20261024/verify_current_main.py). It independently validates sequence/hash pairing, all 9,000 real inputs, non-silent audio, clean teardown and the reported percentiles. The following hashes refer to decompressed original records:

| Raw record | SHA256 |
| --- | --- |
| `validated-run/base/run-1.json` | `739454e3b57ed1977cb2fc9529cc9d5d2163b0afcfcd053bed30ab611fc0839a` |
| `validated-run/base/run-2.json` | `1f75ed52502552343546281a9d4a2166de2364f3d7642b525ffac7679023c51b` |
| `validated-run/base/run-3.json` | `1c40893b24f4677a176c11bbbfa04b80299377db15d12c324adcb75174d411c2` |
| `validated-run/fix/run-1.json` | `5f82f3366f6b8de8b4136e88a3f11f4265fcc77f7a6f343b1d969354166c246a` |
| `validated-run/fix/run-2.json` | `4da2f4f8783b4f5d48da9cb3465833d7f2b94b52bce2421ad4e6ea5eb161f61f` |
| `validated-run/fix/run-3.json` | `9c590ac299323e533f1b0ebf0ce7b5846f1f278bc8614c05d6b4ac88f8cf1a99` |
