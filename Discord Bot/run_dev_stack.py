from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRINT_LOCK = threading.Lock()


def _print(prefix: str, text: str) -> None:
    with PRINT_LOCK:
        print(f"[{prefix}] {text}", flush=True)


def _stream_output(prefix: str, proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            if line:
                _print(prefix, line)
    except Exception as exc:
        _print(prefix, f"(output reader error: {exc})")


def _request_graceful_stop(proc: subprocess.Popen[str] | None, *, name: str, timeout: float = 8.0) -> bool:
    if proc is None or proc.poll() is not None:
        return True
    try:
        if os.name == "nt":
            # Prefer CTRL_BREAK_EVENT on Windows: it targets the child process group without interrupting this launcher.
            sent = False
            for sig_name in ("CTRL_BREAK_EVENT", "CTRL_C_EVENT"):
                sig = getattr(signal, sig_name, None)
                if sig is None:
                    continue
                try:
                    proc.send_signal(sig)
                    sent = True
                    _print("runner", f"sent {sig_name} to {name} (pid={proc.pid})")
                    break
                except Exception:
                    continue
            if not sent:
                with contextlib.suppress(Exception):
                    proc.terminate()
        else:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        deadline = time.monotonic() + max(0.1, timeout)
        while True:
            code = proc.poll()
            if code is not None:
                _print("runner", f"{name} exited with code {code}")
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                proc.wait(timeout=min(0.5, remaining))
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                # Windows can reflect CTRL_BREAK to the parent console as well.
                # Keep waiting for child shutdown instead of treating it as failure.
                _print("runner", f"ignoring local interrupt while waiting for {name} to exit")
                continue
    except KeyboardInterrupt:
        # If the user presses Ctrl+C again while we're already shutting down, don't crash the launcher.
        _print("runner", f"interrupted while signalling {name}; will continue shutdown")
        return False
    except Exception:
        return False


def _kill_process_tree(proc: subprocess.Popen[str] | None, *, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        _print("runner", f"force-killed {name} (pid={proc.pid})")
    except Exception:
        with contextlib.suppress(Exception):
            proc.terminate()
            _print("runner", f"terminated {name} (pid={proc.pid})")


def _stop_process(proc: subprocess.Popen[str] | None, *, name: str, timeout: float = 8.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    if _request_graceful_stop(proc, name=name, timeout=timeout):
        return
    _print("runner", f"{name} did not stop gracefully, forcing shutdown...")
    _kill_process_tree(proc, name=name)


def _spawn(name: str, args: list[str]) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if name == "bot":
        # bot.py enforces interactive TTY by default; the launcher pipes stdout to multiplex logs
        env.setdefault("LIVE_ROLE_BOT_ALLOW_NONINTERACTIVE", "1")

    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        preexec_fn = os.setsid

    _print("runner", f"starting {name}: {' '.join(args)}")
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
        preexec_fn=preexec_fn,
    )


def main() -> int:
    default_agent_mode = "start" if os.name == "nt" else "dev"
    parser = argparse.ArgumentParser(
        description="Run LiveKit agent and Discord bot in one terminal with prefixed logs."
    )
    parser.add_argument(
        "--agent-mode",
        default=default_agent_mode,
        choices=["dev", "start"],
        help=f"Mode passed to livekit_agent.py (default: {default_agent_mode})",
    )
    args = parser.parse_args()

    py = sys.executable
    agent_cmd = [py, "-u", "livekit_agent.py", args.agent_mode]
    bot_cmd = [py, "-u", "bot.py"]

    agent_proc: subprocess.Popen[str] | None = None
    bot_proc: subprocess.Popen[str] | None = None
    readers: list[threading.Thread] = []

    try:
        agent_proc = _spawn("agent", agent_cmd)
        readers.append(threading.Thread(target=_stream_output, args=("agent", agent_proc), daemon=True))
        readers[-1].start()

        # Give agent a moment to boot and register before starting the Discord bot.
        time.sleep(1.0)

        bot_proc = _spawn("bot", bot_cmd)
        readers.append(threading.Thread(target=_stream_output, args=("bot", bot_proc), daemon=True))
        readers[-1].start()

        while True:
            agent_code = agent_proc.poll()
            bot_code = bot_proc.poll()
            if agent_code is not None:
                _print("runner", f"agent exited with code {agent_code}, stopping bot...")
                _stop_process(bot_proc, name="bot", timeout=10.0)
                return agent_code or 0
            if bot_code is not None:
                _print("runner", f"bot exited with code {bot_code}, stopping agent...")
                _stop_process(agent_proc, name="agent", timeout=10.0)
                return bot_code or 0
            time.sleep(0.25)
    except KeyboardInterrupt:
        _print("runner", "Ctrl+C received, stopping agent and bot...")
        # Stop the Discord bot first so it can leave voice channels cleanly.
        with contextlib.suppress(KeyboardInterrupt):
            _stop_process(bot_proc, name="bot", timeout=12.0)
        with contextlib.suppress(KeyboardInterrupt):
            _stop_process(agent_proc, name="agent", timeout=12.0)
        return 130
    finally:
        with contextlib.suppress(KeyboardInterrupt):
            _stop_process(bot_proc, name="bot", timeout=2.0)
        with contextlib.suppress(KeyboardInterrupt):
            _stop_process(agent_proc, name="agent", timeout=2.0)
        for t in readers:
            t.join(timeout=0.5)


if __name__ == "__main__":
    raise SystemExit(main())
