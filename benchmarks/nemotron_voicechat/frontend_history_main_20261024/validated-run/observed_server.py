"""Run the stock CLI with a common, read-only append and PCM observation ledger."""

from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import os
import threading
import time
from functools import wraps
from pathlib import Path


def install_observation() -> None:
    from vllm_omni.engine.duplex.contracts import duplex_session_id_from_request_id
    from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
    from vllm_omni.model_executor.models.nemotron_voicechat.duplex.data_plane import (
        NemotronVoiceChatDataPlaneSession,
        _multimodal,
        _unwrap,
    )
    from vllm_omni.model_executor.models.nemotron_voicechat.duplex.plugin import (
        NemotronVoiceChatDuplexPlugin,
    )

    ledger = Path(os.environ["VOICECHAT_OBSERVATION_LEDGER"]).open("x", buffering=1)
    lock = threading.Lock()
    automatic = contextvars.ContextVar("observed_automatic_silence", default=None)

    def emit(kind: str, **values: object) -> None:
        record = json.dumps(
            {"kind": kind, "observed_ns": time.monotonic_ns(), **values}, default=str
        )
        with lock:
            ledger.write(record + "\n")

    original_start = DuplexSessionRunner._start_append

    @wraps(original_start)
    async def observed_start(self, payload, **kwargs):
        token = automatic.set(bool(kwargs.get("silence_continuation", False)))
        try:
            # The original creates an asyncio Task, which inherits this context.
            # Its payload, scheduling, callbacks, and return value are unchanged.
            return await original_start(self, payload, **kwargs)
        finally:
            automatic.reset(token)

    original_plan = NemotronVoiceChatDuplexPlugin.plan_append

    @wraps(original_plan)
    def observed_plan(self, **kwargs):
        plan = original_plan(self, **kwargs)
        raw = base64.b64decode(kwargs["payload"]["audio"], validate=True)
        emit(
            "append",
            session_id=kwargs["fence"].session_id,
            request_id=kwargs["request_id"],
            epoch=kwargs["fence"].epoch,
            model_seq=kwargs["seq"],
            automatic_silence=automatic.get(),
            payload_sha256=hashlib.sha256(raw).hexdigest(),
            payload_bytes=len(raw),
        )
        return plan

    original_project = NemotronVoiceChatDataPlaneSession.project_output

    @wraps(original_project)
    def observed_project(self, output, **kwargs):
        inner, completion, stage_id = _unwrap(output)
        request_id = str(
            getattr(inner, "request_id", None) or getattr(output, "request_id", "")
        )
        if stage_id == 0:
            metadata = _multimodal(output, completion)
            emit(
                "thinker",
                session_id=duplex_session_id_from_request_id(request_id),
                request_id=request_id,
                meta=metadata.get("meta"),
                token_ids=metadata.get("nvc_text_token_ids"),
            )
        for projected in original_project(self, output, **kwargs):
            encoded = projected.get("audio_data")
            if encoded:
                raw = base64.b64decode(str(encoded).split(",")[-1], validate=True)
                emit(
                    "pcm",
                    session_id=duplex_session_id_from_request_id(request_id),
                    request_id=request_id,
                    model_seq=self._requests[request_id].audio_frames,
                    pcm_sha256=hashlib.sha256(raw).hexdigest(),
                    pcm_bytes=len(raw),
                    sample_rate_hz=projected["sample_rate_hz"],
                    audio_duration_ms=projected["audio_duration_ms"],
                )
            yield projected

    DuplexSessionRunner._start_append = observed_start
    NemotronVoiceChatDuplexPlugin.plan_append = observed_plan
    NemotronVoiceChatDataPlaneSession.project_output = observed_project
    emit("installed", process_id=os.getpid())


if __name__ == "__main__":
    install_observation()
    from vllm_omni.entrypoints.cli.main import main

    main()
