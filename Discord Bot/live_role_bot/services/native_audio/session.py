from __future__ import annotations

import asyncio
import contextlib
import logging

from google.genai import types

from .config import _build_live_connect_config, _build_system_instruction
from .state import _NativeSessionState

logger = logging.getLogger("gemini_native_audio")


class _NativeAudioSessionMixin:
    def _build_system_instruction(self, prompt: str, preferred_language: str) -> str:
        return _build_system_instruction(prompt, preferred_language)

    def _build_config(self, prompt: str, preferred_language: str) -> types.LiveConnectConfig:
        return _build_live_connect_config(
            prompt=prompt,
            preferred_language=preferred_language,
            voice_name=self.voice_name,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    async def _run_session(
        self,
        state: _NativeSessionState,
        system_prompt: str,
        preferred_language: str,
    ) -> None:
        last_error: Exception | None = None
        try:
            model_order = await self._resolve_connect_models()
            logger.info("Native audio model order: %s", ", ".join(model_order))
            for model in model_order:
                try:
                    await self._run_session_once(state, system_prompt, preferred_language, model)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_error = exc
                    reason = str(exc).lower()
                    if "operation is not implemented" in reason or "not supported" in reason:
                        # Avoid repeatedly selecting a model that rejects live operations.
                        if self._resolved_model_order is not None:
                            self._resolved_model_order = [item for item in self._resolved_model_order if item != model]
                            if not self._resolved_model_order:
                                self._resolved_model_order = [m for m in self.models if m != model]
                        logger.warning("Native audio model rejected live ops; disabling model=%s", model)
                    logger.warning("Native audio connect failed for model=%s: %s", model, exc)
                    continue

            if last_error is not None:
                raise last_error
            raise RuntimeError("No Gemini native audio model connected")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Native audio session failed for guild=%s: %s", state.guild_id, exc)
            state.ready_error = f"Gemini Native Audio error: {type(exc).__name__}"
            await self._send_debug(state, f"Gemini Native Audio error: `{type(exc).__name__}`")
        finally:
            state.stop_event.set()
            await self._interrupt_playback(state.guild_id)
            if not state.ready_event.is_set():
                state.ready_event.set()

    async def _run_session_once(
        self,
        state: _NativeSessionState,
        system_prompt: str,
        preferred_language: str,
        model: str,
    ) -> None:
        config = self._build_config(system_prompt, preferred_language)
        async with self._api_client.aio.live.connect(model=model, config=config) as session:
            logger.info("Native audio connected: guild=%s model=%s", state.guild_id, model)
            if not state.ready_event.is_set():
                state.ready_event.set()

            sender = asyncio.create_task(self._sender_loop(state, session), name=f"native-sender-{state.guild_id}")
            receiver = asyncio.create_task(self._receiver_loop(state, session), name=f"native-receiver-{state.guild_id}")
            player = asyncio.create_task(self._playback_loop(state), name=f"native-player-{state.guild_id}")
            stopper = asyncio.create_task(state.stop_event.wait(), name=f"native-stop-{state.guild_id}")

            done, pending = await asyncio.wait(
                {sender, receiver, player, stopper},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            for task in done:
                if task is stopper:
                    continue
                with contextlib.suppress(asyncio.CancelledError):
                    err = task.exception()
                    if err:
                        raise err
