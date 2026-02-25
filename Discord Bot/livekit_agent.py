import sys

from live_role_bot.services.livekit.agent_runtime import main


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        text = str(exc)
        if "LIVEKIT_ENABLED is false" in text:
            print(
                "LiveKit runtime is disabled. Set LIVEKIT_ENABLED=true and configure LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if text.startswith("LIVEKIT_") or "GOOGLE_API_KEY" in text or "GEMINI_API_KEY" in text:
            print(f"LiveKit runtime config error: {text}", file=sys.stderr)
            print(
                "Fill LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET (and GOOGLE_API_KEY or GEMINI_API_KEY) in .env.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        raise
