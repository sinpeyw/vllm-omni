# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in validation observer, activated only by the documented PYTHONPATH.

Wait for the native thinker import, then observe encoder batch sizes and final
session cleanup. No GPU synchronization or replacement scheduler is introduced.
"""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path

TARGET = "vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker"
DESTINATION = os.environ.get("VOICECHAT_PERCEPTION_TRACE")


def record(payload):
    with Path(DESTINATION).open("a") as stream:
        stream.write(json.dumps({"time_ns": time.monotonic_ns(), "pid": os.getpid(), **payload}) + "\n")


def observe(module):
    cls = module.NemotronVoiceChatThinkerForConditionalGeneration
    initialize = cls.__init__

    @functools.wraps(initialize)
    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        step = self.perception.encoder.cache_aware_stream_step

        @functools.wraps(step)
        def measured(*step_args, **step_kwargs):
            result = step(*step_args, **step_kwargs)
            signal = step_kwargs["processed_signal"]
            record(
                {
                    "kind": "perception",
                    "batch": signal.shape[0],
                    "shape": list(signal.shape),
                    "drop": step_kwargs["drop_extra_pre_encoded"],
                    "dtype": str(signal.dtype),
                }
            )
            return result

        self.perception.encoder.cache_aware_stream_step = measured

    finished = cls.on_requests_finished

    @functools.wraps(finished)
    def finish(self, request_ids):
        result = finished(self, request_ids)
        record({"kind": "finished", "request_ids": sorted(request_ids), "remaining": sorted(self._sessions)})
        return result

    cls.__init__ = init
    cls.on_requests_finished = finish
    record({"kind": "observer", "model_source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()})


class ObservedLoader(importlib.abc.Loader):
    """Delegate normal source loading, then install the validation wrappers."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        observe(module)


class Observer(importlib.abc.MetaPathFinder):
    """Intercept only the exact thinker module, leaving other imports intact."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = ObservedLoader(spec.loader)
        return spec


if DESTINATION:
    sys.meta_path.insert(0, Observer())
