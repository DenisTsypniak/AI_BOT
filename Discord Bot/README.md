# Discord AI Bot (Ollama + Memory + Voice)

This bot is a standalone Discord bot that:
- keeps contextual dialogue per channel;
- stores memory in SQLite;
- supports a configurable behavior prompt per server;
- answers questions using local Ollama;
- can join voice channels and speak responses.

## Features
- `Context memory`: stores user/assistant messages in `SQLite`.
- `Prompt control`: set/reset prompt for a whole server.
- `Context-aware responses`: each reply uses recent channel history.
- `Voice mode in calls`: join voice channel and speak generated replies.
- `Auto-reply rules`: mention-only mode, allowlisted channels, or all channels.

## Requirements
- Python 3.11+
- `ffmpeg` in PATH (required for voice playback)
- Running Ollama server (`http://127.0.0.1:11434` by default)
- Discord bot token

## Setup
1. Install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Create environment file:
   ```powershell
   Copy-Item .env.example .env
   ```
3. Fill `.env`:
- `DISCORD_TOKEN`
- `OLLAMA_MODEL` (example: `llama3.1:8b`)
- optionally prompt and voice settings
4. Start bot:
   ```powershell
   python bot.py
   ```

## Bot Commands
- `!help_ai` Show all commands.
- `!ask <text>` Ask the assistant.
- `!prompt_set <text>` Set behavior prompt for current server (Manage Server).
- `!prompt_show` Show active behavior prompt.
- `!prompt_reset` Reset prompt to default (Manage Server).
- `!memory_clear` Clear channel memory (Manage Messages).
- `!join` Join your current voice channel.
- `!leave` Leave voice channel.
- `!voice_on` Enable auto-voice replies in server.
- `!voice_off` Disable auto-voice replies in server.
- `!say <text>` Speak text in voice channel.

## Important Notes
- Use official Discord bot account only.
- The bot can speak in voice channels now.
- Full speech-to-speech (listen to user voice and respond automatically) is a separate stage and can be added with STT pipeline.
