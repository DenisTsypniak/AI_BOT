from __future__ import annotations

from .message_mixin import MessageMixin
from .prompt_mixin import PromptMixin


class DialogueMixin(
    MessageMixin,
    PromptMixin,
):
    pass
