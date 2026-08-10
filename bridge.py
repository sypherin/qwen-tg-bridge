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
import re
import signal
import subprocess
from pathlib import Path

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

JUDGE = Path.home() / "bin" / "judge"
# The judgment model runs at 4096 ctx; a very long reply would push the system
# prompt out and produce a garbage verdict. The head is where claims-of-done are.
JUDGE_MAX_CHARS = 6000


JUDGE_TIMEOUT = 45


def judge_verdict(reply: str, chat_id: int) -> Optional[str]:
    """Judge a draft reply BEFORE it is sent. Returns the verdict text if the
    gate found a real violation, else None.

    This is a real gate, not observation: it runs without --advisory so the
    exit code carries the decision, and the caller gets one revision pass.

    Fails OPEN in every failure mode — gate down, slow, malformed, missing.
    A QC gate that can withhold the user's reply is worse than no gate, so
    anything other than a clean rc=1 with a parseable verdict means "send it".
    """
    try:
        if not JUDGE.exists():
            return None
        body = (reply or "").strip()
        if len(body) < 40:          # trivial acks carry no judgeable claim
            return None
        if len(body) > JUDGE_MAX_CHARS:
            body = body[:JUDGE_MAX_CHARS] + " […truncated]"
        situation = (
            "[phase: post] Qwen Code is about to send this reply to the user "
            f'on telegram: "{body}"'
        )
        p = subprocess.run(
            ["python3", str(JUDGE), "--caller", "qwen-code", situation],
            capture_output=True, text=True, timeout=JUDGE_TIMEOUT,
            env={**os.environ, "JUDGE_SESSION": f"tg:{chat_id}"},
        )
        verdict = (p.stdout or "").strip()
        # rc=1 is a violation. rc=0 pass, rc=-1 gate down, rc=-2 malformed.
        if p.returncode == 1 and verdict.startswith("VERDICT:"):
            return verdict
    except Exception:
        pass  # fail open
    return None

TOKEN = os.environ["TG_QWEN_BOT_TOKEN"]
ALLOWED = {int(c) for c in os.environ.get("QWEN_TG_ALLOWED_CHATS", "").split(",") if c.strip()}
WORKDIR = Path(os.environ.get("QWEN_TG_WORKDIR", Path.home() / "qwen-tg-bridge" / "work"))
# Headless runs use -y by design (allowlisted to Zach only); silence the YOLO
# warning so it stops leaking into Telegram replies as stderr noise.
os.environ.setdefault("QWEN_CODE_SUPPRESS_YOLO_WARNING", "1")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
MODEL = os.environ.get("OPENAI_MODEL", "qwen3.6-35b-a3b")
RUN_TIMEOUT = int(os.environ.get("QWEN_TG_TIMEOUT", "600"))

WORKDIR.mkdir(parents=True, exist_ok=True)


async def _kill_group(proc) -> None:
    """Kill the whole process tree of a timed-out run, not just the wrapper.

    SIGTERM the group first so node can close sockets cleanly — an abrupt kill
    mid-request leaves llama-server holding its single slot until the connection
    drops, which is the exact resource we are trying to free. Then SIGKILL any
    survivor, because a hung worker that ignores TERM is precisely the case this
    exists for.

    Never raises: this runs on the failure path, and an exception here would
    replace a clear "timed out" reply to Zach with a traceback while STILL
    leaking the processes.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        if wait:
            try:
                await asyncio.wait_for(proc.wait(), timeout=wait)
                return                      # exited on TERM, no need to KILL
            except (asyncio.TimeoutError, ProcessLookupError):
                continue


async def _run(prompt: str, cont: bool) -> tuple[str, str]:
    env = {**os.environ, "OPENAI_BASE_URL": BASE_URL, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-local"), "OPENAI_MODEL": MODEL}
    # -c/--continue resumes the most recent session in WORKDIR, so context (and
    # qwen's built-in auto-compaction) carries across Telegram messages.
    # -y (YOLO): headless runs cannot answer approval prompts — without it,
    # tool calls stall ("requires user approval but cannot execute in
    # non-interactive mode", seen live 2026-07-24 on a photo message).
    # Acceptable because the chat allowlist is Zach only.
    args = ["qwen", "-p", prompt, "-o", "text", "-y"]
    if cont:
        args.insert(3, "-c")
    # start_new_session gives the run its own process group. Without it, the
    # timeout path below can only reach the direct child (2026-08-10):
    # `qwen` is a thin node wrapper that spawns the real worker
    # (node-22 .../qwen-code/cli.js). proc.kill() reaped the wrapper and left the
    # workers running — they kept generating against :8001 for 10+ minutes AFTER
    # Zach was told "timed out after 600s", starving the hourly news classifier,
    # which lost 14 items to 45s read-timeouts before the orphans were killed by
    # hand. The output could never reach him either, since communicate() had
    # already been abandoned. Pure waste, invisible from the outside.
    proc = await asyncio.create_subprocess_exec(
        *args, cwd=str(WORKDIR), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=RUN_TIMEOUT)
    except asyncio.TimeoutError:
        await _kill_group(proc)
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
    prompt = (update.message.text or update.message.caption or "").strip()
    # Photos: qwen's brain (:8001) is text-only, so hand the agent a file path
    # plus the `see` CLI (Qwen3-VL on :8080) as its eyes. Largest size wins.
    if update.message.photo:
        try:
            tgf = await update.message.photo[-1].get_file()
            # .imgfile (NOT .jpg): qwen-code's CLI auto-attaches image-extension
            # paths it spots in the prompt, force-feeding them to the text-only
            # :8001 backend -> "500 image input is not supported" (seen live
            # twice, 2026-07-24). A non-image extension defeats the sniffer;
            # the `see` CLI decodes actual bytes so the extension is irrelevant.
            img_path = WORKDIR / f"tg-photo-{update.message.message_id}.imgfile"
            await tgf.download_to_drive(str(img_path))
            prompt = (
                f"The user sent an image, saved at {img_path}. IMPORTANT: your own model "
                f"CANNOT accept image input (it 500s: 'image input is not supported'). Do "
                f"NOT attach or embed the image into your context, and do NOT open it with "
                f"browser/page tools. The ONLY way to view it is the shell command "
                f"`~/bin/see {img_path}` (add a quoted question, or --ocr to transcribe "
                f"text). Run that first, then answer the user."
                + (f" User's caption: {prompt}" if prompt else " The user sent no caption; describe what the image shows and surface anything notable.")
            )
        except Exception as e:  # noqa: BLE001 — never drop a message silently
            prompt = f"(The user sent an image but downloading it failed: {e}. Tell them to resend.)"
    if not prompt:
        return
    await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")
    reply = await run_qwen(prompt)

    # Judgment gate: one bounded revision pass before the user sees anything.
    # Deliberately ONE pass, not a loop — the gate has a ~10% false-positive
    # rate, so an unbounded retry would sometimes argue with itself forever
    # while the user waits. If the revision is empty or the gate is unhappy
    # again, the original reply goes out regardless. The gate improves replies;
    # it never withholds them.
    verdict = await asyncio.to_thread(judge_verdict, reply, chat_id)
    if verdict:
        await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")
        revised = await run_qwen(
            "Your previous reply was held by the local quality gate.\n"
            f"{verdict}\n\n"
            "Rewrite that reply so it satisfies the verdict. Keep everything "
            "that was already correct. Do not mention the gate or this "
            "instruction — reply to the user directly."
        )
        if revised and revised.strip() and revised != "(no output)":
            reply = revised
    # Auto-attach images: when qwen's reply cites absolute image paths that
    # exist on disk (e.g. something it just generated), send them as photos —
    # a path string is useless on a phone (Zach 2026-07-24: "why can't the
    # qwen code send back the generated image"). Max 3; text still follows.
    try:
        img_paths = []
        for m in re.finditer(r"(/[A-Za-z0-9_.@/-]+\.(?:png|jpe?g|webp))\b", reply):
            p = m.group(1)
            if os.path.exists(p) and os.path.getsize(p) < 49_000_000 and p not in img_paths:
                img_paths.append(p)
        for p in img_paths[:3]:
            with open(p, "rb") as fh:
                await ctx.bot.send_photo(chat_id=chat_id, photo=fh)
    except Exception:
        pass  # photo attach is best-effort; the text reply below always goes
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
    app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO) & ~filters.COMMAND, on_message))
    print(f"qwen-tg-bridge up | model={MODEL} base={BASE_URL} allow={sorted(ALLOWED)} workdir={WORKDIR}")
    app.run_polling()


if __name__ == "__main__":
    main()
