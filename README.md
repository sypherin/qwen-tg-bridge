# qwen-tg-bridge

Drive [Qwen Code](https://github.com/QwenLM/qwen-code) (the Qwen coding CLI) from Telegram — against a local model or any OpenAI-compatible endpoint. Send a prompt from your phone; the bridge runs the agent headless in a working directory and streams live progress back, then the reply.

Built for a home-lab box running llama-server; works with vLLM, LM Studio, Ollama's OpenAI API, or Qwen's own API.

## What it does

- **Runs Qwen Code headless** (`qwen -p … -o stream-json -y`) in a sandboxed workdir, one run at a time
- **Live progress** — parses the stream-json events and edits a Telegram status bubble with the current reasoning tail, tool calls, and forming answer; everything also tees into a tmux pane you can attach to
- **Voice notes** — transcribed locally with faster-whisper (`hear.py`, CPU int8, no API); the transcript is echoed back before the run so a mishear is visible immediately
- **Photos** — downloaded and handed to the agent as a file path with an OCR/vision helper CLI (your model may be text-only; the prompt tells the agent how to look without embedding the image)
- **Documents** — PDFs, code, data files land in the workdir for the agent to read with its own tools
- **Run-and-fix loop** (`verify.py`) — after a coding run, smoke-test what was actually produced (execute it / render it headless) and loop the model back on real runtime failures. Fails open: a broken harness never blocks a reply
- **Optional quality gate** — if `~/bin/judge` exists, replies pass a local LLM judgment gate with one bounded revision pass. Fails open by design
- **Safety rails** — chat-ID allowlist (refuses to start without one), stall timeout (dies only after N seconds of *silence*, not wall-clock), hard ceiling, heartbeat pings, `/stop`, stale-message replay guard, whole-process-group kills (a killed run never leaves orphan workers hammering your endpoint)

## Commands

`/new` fresh session · `/stop` kill the running job · `/status` model routing + run state · `/retry` re-run last prompt · `/think on|off` reasoning toggle (via a `-think` provider id) · `/fast on|off` route simple tasks to a second, cheaper model

Every reply carries a footer: `— model · wall time · tool count`.

## Quick start

```bash
git clone https://github.com/sypherin/qwen-tg-bridge && cd qwen-tg-bridge
pip install -r requirements.txt          # python-telegram-bot, faster-whisper

cp .env.example ~/.env.qwen-tg-bridge    # fill in your bot token
./start.sh                               # first run captures your chat_id from a message you send the bot
```

Requirements: Python 3.11+, the `qwen` CLI installed (`npm i -g @qwen-code/qwen-code`), an OpenAI-compatible endpoint.

For always-on: `tmux-start.sh` runs the bridge inside a tmux session (attach with `tmux attach -t qwentg`) and restarts it on crash; wire it to systemd as a oneshot, or just use `run-in-tmux.sh` directly.

## Configuration

Environment (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `TG_QWEN_BOT_TOKEN` | — | required, from @BotFather |
| `QWEN_TG_ALLOWED_CHATS` | — | required, comma-separated numeric chat IDs |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8022/v1` | your endpoint |
| `OPENAI_MODEL` | `qwen3.8-27b` | default model id |
| `QWEN_TG_WORKDIR` | `~/qwen-tg-bridge/work` | agent sandbox |
| `QWEN_TG_IDLE_TIMEOUT` | `1800` | seconds of silence before a run is killed (0 = off) |
| `QWEN_TG_MAX_TIMEOUT` | `10800` | absolute ceiling |
| `QWEN_TG_HEARTBEAT` | `1200` | "still working" ping interval |
| `QWEN_TG_PROGRESS_EVERY` | `8` | min seconds between status-bubble edits |
| `QWEN_TG_VERIFY` | `1` | run-and-fix loop |
| `QWEN_TG_MAX_FIX_ROUNDS` | `2` | fix-loop bound |
| `QWEN_TG_STALE_AFTER` | `1200` | drop queued messages older than this on restart |

## Design notes

- **Allowlist-only.** The bridge runs the agent with `-y` (auto-approve) — that is only acceptable because it replies exclusively to allowlisted chat IDs. Do not remove the allowlist check.
- **Single slot.** One run at a time; a second prompt is told to wait or `/stop`. If your endpoint can take parallel sessions, the lock is the thing to relax.
- **Edited messages are not re-run** (an edit would redo a coding run's file writes); the bot says so instead of staying silent.
- **Observable by default.** Live status bubble, heartbeat, run footer, tmux scrollback. A long agent run should never be indistinguishable from a hang.

## License

MIT
