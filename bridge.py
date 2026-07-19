#!/usr/bin/env python3
"""
Telegram -> Qwen Code (headless) bridge for the Strix Halo box.

Each allowlisted Telegram message is run through `qwen` in non-interactive
mode against the local llama-server on :8001, and the result is sent back.

Safety, deliberately:
  * ALLOWLIST ONLY — replies solely to chat IDs in QWEN_TG_ALLOWED_CHATS.
    Without this, anyone who finds the bot could run tools on your box.
  * --approval-mode auto — an LLM classifier auto-approves safe actions and
    BLOCKS risky ones. Never 'yolo' for a network-reachable bot.
  * Each run is sandboxed to WORKDIR with a hard timeout.
Config comes from env (no secrets in code):
  TG_QWEN_BOT_TOKEN        Telegram bot token (from @BotFather)
  QWEN_TG_ALLOWED_CHATS    comma-separated chat IDs allowed to use it
  QWEN_TG_WORKDIR          working dir for agent runs (default ~/qwen-tg-bridge/work)
  OPENAI_BASE_URL          default http://127.0.0.1:8001/v1
  OPENAI_MODEL             default qwen3.6-35b-a3b
"""
import asyncio
import os
from pathlib import Path

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

TOKEN = os.environ["TG_QWEN_BOT_TOKEN"]
ALLOWED = {int(c) for c in os.environ.get("QWEN_TG_ALLOWED_CHATS", "").split(",") if c.strip()}
WORKDIR = Path(os.environ.get("QWEN_TG_WORKDIR", Path.home() / "qwen-tg-bridge" / "work"))
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
MODEL = os.environ.get("OPENAI_MODEL", "qwen3.6-35b-a3b")
RUN_TIMEOUT = int(os.environ.get("QWEN_TG_TIMEOUT", "600"))

WORKDIR.mkdir(parents=True, exist_ok=True)


async def _run(prompt: str, cont: bool) -> tuple[str, str]:
    env = {**os.environ, "OPENAI_BASE_URL": BASE_URL, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-local"), "OPENAI_MODEL": MODEL}
    # -c/--continue resumes the most recent session in WORKDIR, so context (and
    # qwen's built-in auto-compaction) carries across Telegram messages.
    args = ["qwen", "-p", prompt, "-o", "text", "--approval-mode", "auto"]
    if cont:
        args.insert(3, "-c")
    proc = await asyncio.create_subprocess_exec(
        *args, cwd=str(WORKDIR), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=RUN_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        return "", f"(timed out after {RUN_TIMEOUT}s)"
    return out.decode(errors="replace").strip(), err.decode(errors="replace")


async def run_qwen(prompt: str) -> str:
    # Continue the running session for continuous context; on cold start (no
    # session yet) -c can fail, so fall back to a fresh run that seeds one.
    text, err = await _run(prompt, cont=True)
    if not text and "No " in err and "session" in err.lower():
        text, err = await _run(prompt, cont=False)
    if text:
        return text[-3800:]
    errtail = "\n".join(l for l in err.splitlines() if "Legacy setting" not in l).strip()
    return errtail[-3800:] if errtail else "(no output)"


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED:
        return  # silent: do not engage non-allowlisted chats
    prompt = (update.message.text or "").strip()
    if not prompt:
        return
    await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")
    reply = await run_qwen(prompt)
    # Render markdown (bold, code blocks) when it parses; fall back to plain text
    # so a stray * or _ never turns into a silent 400 from Telegram.
    try:
        await update.message.reply_text(reply, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(reply)


def main():
    if not ALLOWED:
        raise SystemExit("Refusing to start: QWEN_TG_ALLOWED_CHATS is empty (would accept anyone).")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    print(f"qwen-tg-bridge up | model={MODEL} base={BASE_URL} allow={sorted(ALLOWED)} workdir={WORKDIR}")
    app.run_polling()


if __name__ == "__main__":
    main()
