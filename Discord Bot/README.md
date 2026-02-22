# Discord Live Role Bot

A rewritten Discord bot focused on:
- live role-based dialogue;
- persistent per-user memory;
- rolling summaries per user/channel;
- Gemini 2.5 Flash Native Audio Dialog voice sessions;
- adaptive responses that use stored memory facts.

## Project Structure
Main implementation now lives in a package layout:

```text
live_role_bot/
  app.py                        # startup, wiring, logging, main()
  config.py                     # environment settings + validation
  discord/
    client.py                   # Discord client core lifecycle
    common.py                   # shared helpers + dataclasses
    voice_integration.py        # voice_recv integration + sink/guard
    mixins/
      identity_mixin.py         # identity/role resolution + dedupe
      prompt_mixin.py           # context + system prompt construction
      message_mixin.py          # text/native transcript flows + command handling
      workers_mixin.py          # profile/summary async workers
      voice_mixin.py            # voice capture and STT/native-audio hooks
      dialogue_mixin.py         # prompt + message composition
      workers_voice_mixin.py    # worker + voice composition
  services/
    gemini_client.py
    local_stt.py
    native_audio/
      manager.py                # public GeminiNativeAudioManager
      base.py                   # session lifecycle + model fallback
      session.py                # live connect / run orchestration
      io.py                     # sender / receiver loops
      playback.py               # audio playback pipeline
      events.py                 # transcript callbacks / debug posts
      audio.py                  # PCM source + conversion helpers
      state.py                  # dataclasses for session state/queues
      config.py                 # live config + instruction builder
  memory/
    extractor.py
    store.py                    # composed public MemoryStore
    storage/
      schema.py                 # schema + migration reset/init
      identity.py               # users table operations
      roles.py                  # role_profiles + guild_settings
      messages.py               # sessions/messages/history
      summaries.py              # dialogue summary upserts/lookups
      stt.py                    # stt_turns persistence
      facts.py                  # fact upsert/merge/evidence
      utils.py                  # shared helpers
```

Single supported entrypoint is `bot.py`, which starts `live_role_bot.app`.

## Core Behavior
- Available command: `!join` (prefix is configurable via `DISCORD_COMMAND_PREFIX`).
- In guilds, bot replies when mentioned, in allowlisted channels, or everywhere if mention-only is disabled.
- In DMs, bot always replies.
- When mentioned by a user in voice channel (and voice auto-join enabled), bot starts Gemini Native Audio session and streams audio in real time.

## Memory Model
Data is stored in SQLite tables:
- `users`
- `role_profiles`
- `guild_settings`
- `sessions`
- `messages`
- `stt_turns`
- `user_facts`
- `fact_evidence`
- `dialogue_summaries`

This supports:
- sessioned conversation history;
- durable user fact storage with confidence, importance, evidence count;
- auditable STT quality (when local STT mode is enabled);
- rolling summary updates.

## Setup
1. Create and activate virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Create `.env` from template:
   ```powershell
   Copy-Item .env.example .env
   ```
4. Fill required values:
   - `DISCORD_TOKEN`
   - `GEMINI_API_KEY`
5. Run:
   ```powershell
   python bot.py
   ```

## Required Intents
Enable in Discord Developer Portal:
- Message Content Intent
- Server Members Intent (recommended for identity sync)

## Notes
- Native audio mode requires `google-genai` + `discord-ext-voice_recv`.
- Local STT is optional and only used when Native Audio mode is disabled.
- RP canon/persona can be loaded from `BOT_HISTORY_JSON_PATH` (default `./data/bot_history.json`).
- The bot stores role profile and memory locally in SQLite file configured by `SQLITE_PATH`.
