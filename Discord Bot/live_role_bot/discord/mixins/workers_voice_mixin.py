from __future__ import annotations

from .voice_mixin import VoiceMixin
from .workers_mixin import WorkersMixin


class WorkersVoiceMixin(
    VoiceMixin,
    WorkersMixin,
):
    pass
