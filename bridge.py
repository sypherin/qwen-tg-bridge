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
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters

JUDGE = Path.home() / "bin" / "judge"
# The judgment model runs at 4096 ctx; a very long reply would push the system
# prompt out and produce a garbage verdict. The head is where claims-of-done are.
JUDGE_MAX_CHARS = 6000


JUDGE_TIMEOUT = 45

# Voice notes (2026-09-01): transcribe locally with faster-whisper via hear.py
# (sits beside this file, mirrors ~/bin/see). CPU int8 "small" — a 60s note
# takes ~5-10s; cold model load ~2s. Timeout is generous because the Strix box
# may be busy with a qwen run on the same CPU.
HEAR = Path(__file__).resolve().parent / "hear.py"
HEAR_TIMEOUT = int(os.environ.get("QWEN_TG_HEAR_TIMEOUT", "180"))


def _transcribe(path: Path) -> str:
    """Voice note -> text. Raises on failure; empty string = nothing intelligible."""
    try:
        p = subprocess.run(
            ["python3", str(HEAR), str(path)],
            capture_output=True, text=True, timeout=HEAR_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"transcription timed out after {HEAR_TIMEOUT}s")
    if p.returncode == 0:
        return (p.stdout or "").strip()
    raise RuntimeError((p.stderr or f"exit {p.returncode}").strip()[-300:])


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
# 2026-08-22: a hard wall-clock cap killed long-but-PROGRESSING runs mid-work
# (the pacman run was still iterating when the old 3600s cap axed it). Replaced
# with a STALL timeout: a run only dies after this many seconds with NO output,
# so a run that keeps producing never times out. IDLE_TIMEOUT=0 disables it.
IDLE_TIMEOUT = int(os.environ.get("QWEN_TG_IDLE_TIMEOUT", "1800"))   # 30 min of silence = stuck
# Absolute wall-clock ceiling even for a run that keeps producing output — a
# backstop so a pathological loop can't hold the :8022 slot indefinitely.
# Default 10800s (3h, Zach 2026-08-22); 0 disables it. So a run only ends when it
# finishes, stalls for 30 min, hits 3h, or you /stop it.
MAX_TIMEOUT = int(os.environ.get("QWEN_TG_MAX_TIMEOUT", "10800"))
# Heartbeat: while a run is still going, ping Telegram every this-many seconds so
# a long run (possibly stuck in a loop) surfaces instead of running silently for
# hours — the message reminds Zach it's alive and that /stop exists. 0 disables.
HEARTBEAT = int(os.environ.get("QWEN_TG_HEARTBEAT", "1200"))   # 20 min
# Live progress: while a run streams, edit a Telegram "status" message at most
# this often (seconds) with the latest reasoning tail / tool call. 0 disables.
PROGRESS_EVERY = int(os.environ.get("QWEN_TG_PROGRESS_EVERY", "8"))

# One qwen run at a time (single :8022 slot); /stop kills the live one.
_BUSY = asyncio.Lock()
CURRENT: dict = {"proc": None}

# Reasoning switch (2026-08-22). :8022 toggles thinking per-request via
# chat_template_kwargs.enable_thinking; Qwen Code carries that through a second
# provider id `qwen3.8-27b-think` (extra_body) in ~/.qwen/settings.json. /think
# on|off just picks which -m the bot passes — client-side, so site chat / Deneb
# on :8022 are unaffected. State persists in a file across bridge restarts.
MODEL_THINK = os.environ.get("QWEN_TG_MODEL_THINK", "qwen3.8-27b-think")
MODEL_PLAIN = os.environ.get("QWEN_TG_MODEL_PLAIN", MODEL)
THINK_FILE = Path.home() / "qwen-tg-bridge" / "think_mode"
# Fast lane (2026-09-01, Zach: ":8022 for complex, :8001 for simple"). /fast
# routes runs to the :8001 MoE (qwen3.6-35b provider id in ~/.qwen/settings.json)
# and overrides /think while on — :8001 runs with reasoning off server-side.
MODEL_FAST = os.environ.get("QWEN_TG_MODEL_FAST", "qwen3.6-35b")
FAST_FILE = Path.home() / "qwen-tg-bridge" / "fast_mode"

def _think_on() -> bool:
    try:
        return THINK_FILE.read_text().strip().lower() == "on"
    except Exception:
        return False   # default off (fast)

def _set_think(on: bool) -> None:
    THINK_FILE.write_text("on" if on else "off")

def _fast_on() -> bool:
    try:
        return FAST_FILE.read_text().strip().lower() == "on"
    except Exception:
        return False   # default off (:8022, the stronger model)

def _set_fast(on: bool) -> None:
    FAST_FILE.write_text("on" if on else "off")

# Run-and-fix loop: after a coding run, smoke-test what was produced (execute it / render it
# headless) and loop the model back on real failures — catches runtime bugs that parse clean
# but don't work (e.g. a NaN camera -> blank screen). See verify.py. Off via QWEN_TG_VERIFY=0.
try:
    import verify as _verify  # sits beside this file
except Exception:  # noqa: BLE001 — never let a verify import break the bridge
    _verify = None
VERIFY_ON = os.environ.get("QWEN_TG_VERIFY", "1") != "0" and _verify is not None
MAX_FIX_ROUNDS = int(os.environ.get("QWEN_TG_MAX_FIX_ROUNDS", "2"))

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


async def _run(prompt: str, cont: bool, notify=None, progress=None) -> tuple[str, str]:
    env = {**os.environ, "OPENAI_BASE_URL": BASE_URL, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-local"), "OPENAI_MODEL": MODEL}
    # -c/--continue resumes the most recent session in WORKDIR, so context (and
    # qwen's built-in auto-compaction) carries across Telegram messages.
    # -y (YOLO): headless runs cannot answer approval prompts — without it,
    # tool calls stall ("requires user approval but cannot execute in
    # non-interactive mode", seen live 2026-07-24 on a photo message).
    # Acceptable because the chat allowlist is Zach only.
    # Reasoning on/off = which provider id we select (both point at :8022; the
    # -think one carries extra_body.chat_template_kwargs.enable_thinking).
    model = MODEL_FAST if _fast_on() else (MODEL_THINK if _think_on() else MODEL_PLAIN)
    # -o stream-json (2026-09-01): emits claude-code-style ndjson events on
    # stdout — assistant content blocks (thinking / text / tool_use) plus a
    # final {type:"result"} carrying the reply. We parse them live so the
    # tmux pane shows pretty progress AND Telegram gets edited status updates
    # with the reasoning tail (previously: silent until the final answer).
    args = ["qwen", "-p", prompt, "-o", "stream-json", "-y", "-m", model]
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
    CURRENT["proc"] = proc
    loop = asyncio.get_running_loop()
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    state = {"last": loop.time(), "result": None, "texts": [], "think": "", "last_prog": 0.0, "tools": 0}
    print(f"\n=== run start{' (cont)' if cont else ''} | model={model} ===", flush=True)
    print(f"prompt: {prompt[:200]}{'…' if len(prompt) > 200 else ''}", flush=True)

    def _pane(line: str) -> None:
        # pretty progress → tmux pane (`tmux attach -t qwentg`), best-effort
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception:
            pass  # a display glitch must never kill the run

    async def _progress_tick(label: str, detail: str) -> None:
        # throttled live status to Telegram: latest reasoning tail / tool call
        if not progress or not PROGRESS_EVERY:
            return
        now = loop.time()
        if now - state["last_prog"] < PROGRESS_EVERY:
            return
        state["last_prog"] = now
        try:
            await progress(f"{label} {detail}"[:220] + f" · {int(now - start)}s")
        except Exception:
            pass

    async def _handle_event(j: dict) -> None:
        t = j.get("type")
        if t == "assistant":
            for b in j.get("message", {}).get("content", []) or []:
                bt = b.get("type")
                if bt == "thinking":
                    th = " ".join((b.get("thinking") or "").split())
                    if th:
                        state["think"] = th
                        _pane(f"[think] {th[:180]}")
                        await _progress_tick("🧠", th[-140:])
                elif bt == "text":
                    tx = " ".join((b.get("text") or "").split())
                    if tx:
                        state["texts"].append(tx)
                        _pane(f"[text] {tx[:180]}")
                elif bt == "tool_use":
                    state["tools"] += 1
                    name = b.get("name", "?")
                    try:
                        inp = json.dumps(b.get("input") or {}, ensure_ascii=False)[:100]
                    except Exception:
                        inp = ""
                    _pane(f"[tool] {name} {inp}")
                    await _progress_tick("🔧", f"{name} {inp}"[:140])
        elif t == "result":
            r = j.get("result")
            if isinstance(r, str) and r:
                state["result"] = r
            _pane(f"[done] {j.get('subtype')} turns={j.get('num_turns')}"
                  + (f" tools={state['tools']}" if state["tools"] else ""))

    async def _pump_json(stream, sink):
        # stdout is ndjson events (claude-code-style stream-json). Parse each
        # complete line: pretty-tee to the pane, extract the final result, and
        # surface live reasoning/tool status to Telegram (Zach 2026-09-01: no
        # more silent runs — see what it's thinking while it thinks).
        buf = ""
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            sink.append(chunk)
            state["last"] = loop.time()
            buf += chunk.decode(errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        await _handle_event(json.loads(line))
                        continue
                    except Exception:
                        pass  # not valid JSON after all — fall through as raw
                _pane(f"[raw] {line[:180]}")

    async def _pump_raw(stream, sink):
        # stderr: raw tee only
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            sink.append(chunk)
            state["last"] = loop.time()
            try:
                sys.stdout.write(chunk.decode(errors="replace"))
                sys.stdout.flush()
            except Exception:
                pass  # tee is best-effort; a display glitch must never kill the pump

    pumps = asyncio.gather(_pump_json(proc.stdout, out_chunks), _pump_raw(proc.stderr, err_chunks))
    start = loop.time()
    next_beat = start + HEARTBEAT if (HEARTBEAT and notify) else None
    reason = None
    try:
        while proc.returncode is None:
            await asyncio.sleep(5)
            if proc.returncode is not None:
                break
            now = loop.time()
            idle = now - state["last"]
            if IDLE_TIMEOUT and idle > IDLE_TIMEOUT:
                reason = f"stalled — no output for {int(idle)}s (idle limit {IDLE_TIMEOUT}s)"
                break
            if MAX_TIMEOUT and (now - start) > MAX_TIMEOUT:
                reason = f"hit max runtime {MAX_TIMEOUT}s"
                break
            # Heartbeat: still alive but taking a while — surface it so a possible
            # loop doesn't run silently. Best-effort; a failed ping never aborts.
            if next_beat and now >= next_beat:
                mins = int((now - start) / 60)
                idle_m = int(idle / 60)
                try:
                    await notify(f"⏳ still working — {mins} min elapsed"
                                 + (f", last output {idle_m} min ago" if idle_m >= 2 else "")
                                 + ". send /stop if it's going in circles.")
                except Exception:
                    pass
                next_beat = now + HEARTBEAT
            # Wait-phase ticks (2026-09-01): the qwen CLI sits silent for
            # ~1-4 min during node boot + tool listing before the first event
            # streams. Tick the pane AND the Telegram bubble every 30s so a
            # run is never indistinguishable from a hang (Zach: "i dont see
            # any streaming"). Coordinates with _progress_tick via last_prog.
            if progress and (now - state["last_prog"]) >= 30:
                state["last_prog"] = now
                phase = ("qwen cli spinning up" if not (state["think"] or state["texts"])
                         else "working")
                try:
                    await progress(f"⏳ {phase} · {int(now - start)}s")
                except Exception:
                    pass
                _pane(f"[beat] alive · {int(now - start)}s"
                      + (f" tools={state['tools']}" if state["tools"] else ""))
        if reason:
            await _kill_group(proc)
    finally:
        try:
            await asyncio.wait_for(pumps, timeout=10)
        except Exception:
            pumps.cancel()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:
            pass
        if CURRENT.get("proc") is proc:
            CURRENT["proc"] = None
    # stream-json: the result event carries the final reply; fall back to the
    # accumulated text blocks, then raw stdout (old -o text behaviour).
    out = (state["result"] or "\n".join(state["texts"]).strip()
           or b"".join(out_chunks).decode(errors="replace").strip())
    err = b"".join(err_chunks).decode(errors="replace")
    print(f"\n=== run end rc={proc.returncode} out={len(out)}B err={len(err)}B ===", flush=True)
    # If the group was killed externally (/stop) the loop exits with reason=None
    # but a non-zero/deadly returncode; surface partial output either way.
    if reason:
        err = (err + f"\n({reason})").strip()
    elif proc.returncode and proc.returncode < 0 and not out:
        err = (err + f"\n(stopped)").strip()
    return out, err


async def run_qwen(prompt: str, notify=None, progress=None) -> str:
    # Continue the running session for continuous context; on cold start (no
    # session yet) -c can fail, so fall back to a fresh run that seeds one.
    text, err = await _run(prompt, cont=True, notify=notify, progress=progress)
    if not text and "No " in err and "session" in err.lower():
        text, err = await _run(prompt, cont=False, notify=notify, progress=progress)
    if text:
        return text[-3800:]
    errtail = "\n".join(l for l in err.splitlines() if "Legacy setting" not in l).strip()
    return errtail[-3800:] if errtail else "(no output)"


async def run_qwen_verified(prompt: str, notify=None, progress=None) -> str:
    """run_qwen, then smoke-test what it produced and loop it back to fix real failures.

    Only acts when the run actually changed code files (a chat/question is a no-op). Reports
    failure only on POSITIVE evidence (a runtime error / provably blank canvas), so a broken
    harness never blocks a reply — see verify.py. Bounded by MAX_FIX_ROUNDS.
    """
    if not VERIFY_ON:
        return await run_qwen(prompt, notify=notify, progress=progress)
    before = await asyncio.to_thread(_verify.snapshot, WORKDIR)
    reply = await run_qwen(prompt, notify=notify, progress=progress)
    attempt = 0
    while True:
        after = await asyncio.to_thread(_verify.snapshot, WORKDIR)
        changed = _verify.changed_since(before, after)
        if not changed:
            return reply  # nothing was written — not a coding task
        if attempt == 0 and notify:
            await notify("🔍 running it to check it actually works…")
        res = await asyncio.to_thread(_verify.verify, WORKDIR, changed)
        if res.ok:
            return f"{reply}\n\n✅ auto-verified: {res.summary}"
        if attempt >= MAX_FIX_ROUNDS:
            return (f"{reply}\n\n⚠️ auto-verify still failing after {attempt} fix "
                    f"attempt(s): {res.summary}. Left as-is for you to look at.")
        attempt += 1
        if notify:
            await notify(f"⚙️ auto-check caught an issue — fixing ({attempt}/{MAX_FIX_ROUNDS}): {res.summary}")
        reply = await run_qwen(
            "The code you just produced FAILED an automated smoke-test.\n"
            f"{res.fix_hint}\n\n"
            "Fix it in place in the working directory, then stop. Reply only once it works.",
            notify=notify,
            progress=progress,
        )


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED:
        return  # silent: do not engage non-allowlisted chats
    prompt = (update.message.text or update.message.caption or "").strip()
    # Photos: qwen's brain (:8001) is text-only, so hand the agent a file path
    # plus the `see` CLI (Qwen3.8-27B VL on :8022) as its eyes. Largest size wins.
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
    # Voice/audio notes: transcribe with hear.py and treat the transcript as
    # the prompt. The transcript is echoed back BEFORE the run so a mishear is
    # visible immediately (Zach's observability rule), not after a 1-4 min
    # qwen run built on garbage input.
    va = update.message.voice or update.message.audio
    if va is not None:
        cap = prompt  # audio can carry a caption; voice notes cannot
        try:
            tgf = await va.get_file()
            aud_path = WORKDIR / f"tg-voice-{update.message.message_id}.ogg"
            await tgf.download_to_drive(str(aud_path))
            await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")
            try:
                transcript = await asyncio.to_thread(_transcribe, aud_path)
            except Exception as e:  # noqa: BLE001 — never drop a message silently
                prompt = f"(The user sent a voice message but transcribing it failed: {e}. Tell them to resend as text.)"
            else:
                if not transcript:
                    prompt = ("(The user sent a voice message but the transcript came back empty — "
                              "no intelligible speech. Tell them briefly and suggest typing it.)")
                else:
                    try:
                        await update.message.reply_text(f"🎙 {transcript[:800]}")
                    except Exception:
                        pass  # echo is cosmetic; the run must proceed either way
                    prompt = ("The user sent a voice message; this is its transcript:\n"
                              f"\"\"\"\n{transcript}\n\"\"\"\n"
                              "Treat the transcript as the user's message and reply to it directly."
                              + (f"\n\n(They also added this caption: {cap})" if cap else ""))
        except Exception as e:  # noqa: BLE001 — download failure — never drop silently
            prompt = f"(The user sent a voice message but downloading it failed: {e}. Tell them to resend.)"
    if not prompt:
        return
    # One run at a time — a second prompt while a run is in flight would start a
    # 2nd qwen against the single :8022 slot. Tell the user instead of queueing.
    if _BUSY.locked():
        await update.message.reply_text("⏳ a qwen run is still going — send /stop to cancel it, or wait for it to finish.")
        return
    # Hold the single-run lock across the whole response — the gate revision
    # below also spawns qwen on the same :8022 slot, so it must be serialised too.
    async def _notify(msg: str):
        await ctx.bot.send_message(chat_id=chat_id, text=msg)

    # Live status bubble: created lazily on the first progress event, edited in
    # place (throttled inside _run), deleted once the final reply is imminent.
    prog = {"msg": None, "done": False}

    async def _progress(text: str):
        if prog["done"]:
            return  # run finished — a late tick must never revive the bubble
        try:
            if prog["msg"] is None:
                prog["msg"] = await ctx.bot.send_message(chat_id=chat_id, text=f"⚙️ {text}")
            else:
                await prog["msg"].edit_text(f"⚙️ {text}")
        except Exception:
            pass  # progress is cosmetic — never let it break the run

    async with _BUSY:
        await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")
        # Bubble appears IMMEDIATELY (the CLI's 1-4 min silent boot made
        # "no streaming" indistinguishable from "nothing happened").
        try:
            prog["msg"] = await ctx.bot.send_message(chat_id=chat_id, text="⚙️ qwen starting…")
        except Exception:
            pass
        reply = await run_qwen_verified(prompt, notify=_notify, progress=_progress)
        if not (reply or "").strip() or reply.strip() == "(no output)":
            reply = "(model returned no text — pls send again, or /new if the session is long)"

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
    prog["done"] = True  # freeze the updater BEFORE deleting (late-tick race)
    if prog["msg"] is not None:  # final answer imminent — pop the status bubble
        # Zach 2026-09-01: a flood-controlled delete left the bubble standing
        # with its last thinking snippet as the visible "reply". Retry once,
        # then blank the bubble so stale reasoning is never the final artifact.
        deleted = False
        for _ in range(2):
            try:
                await prog["msg"].delete()
                deleted = True
                break
            except Exception:
                await asyncio.sleep(1.5)
        if not deleted:
            try:
                await prog["msg"].edit_text("✅ done")
            except Exception:
                pass
        prog["msg"] = None
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


async def on_stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/stop — kill the qwen run currently in flight (if any)."""
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED:
        return
    proc = CURRENT.get("proc")
    if proc is not None and proc.returncode is None:
        await _kill_group(proc)
        await update.message.reply_text("🛑 stopped the running qwen job.")
    else:
        await update.message.reply_text("nothing running.")


async def on_think(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/think [on|off] — toggle reasoning for coding runs (default off = faster).
    Client-side (picks the -think model id); does not affect other :8022 users."""
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED:
        return
    arg = ""
    if update.message and update.message.text:
        parts = update.message.text.split()
        arg = parts[1].lower() if len(parts) > 1 else ""
    if arg in ("on", "off"):
        _set_think(arg == "on")
        await update.message.reply_text(
            f"🧠 reasoning {'ON' if arg == 'on' else 'OFF'} for coding runs"
            + (" (slower, better on hard tasks)" if arg == "on" else " (faster)")
        )
    else:
        await update.message.reply_text(
            f"reasoning is currently {'ON' if _think_on() else 'OFF'}. use /think on or /think off."
        )


async def on_fast(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/fast [on|off] — route runs to the :8001 MoE for simple tasks (overrides /think)."""
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED:
        return
    arg = ""
    if update.message and update.message.text:
        parts = update.message.text.split()
        arg = parts[1].lower() if len(parts) > 1 else ""
    if arg in ("on", "off"):
        _set_fast(arg == "on")
        await update.message.reply_text(
            "⚡ fast lane ON: runs go to qwen3.6-35b on :8001 (overrides /think)"
            if arg == "on" else
            "fast lane OFF: runs back on qwen3.8-27b at :8022"
        )
    else:
        await update.message.reply_text(
            f"fast lane is currently {'ON' if _fast_on() else 'OFF'}. use /fast on or /fast off."
        )


def main():
    if not ALLOWED:
        raise SystemExit("Refusing to start: QWEN_TG_ALLOWED_CHATS is empty (would accept anyone).")
    # concurrent_updates=True so /stop is handled WHILE a run is in flight (the
    # default processes updates one at a time, which would queue /stop behind the
    # very run it's meant to cancel). The _BUSY lock still allows only one qwen
    # run at a time; a 2nd prompt gets told to wait or /stop.
    app = ApplicationBuilder().token(TOKEN).concurrent_updates(True).build()
    app.add_handler(CommandHandler("stop", on_stop))
    app.add_handler(CommandHandler("think", on_think))
    app.add_handler(CommandHandler("fast", on_fast))
    app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO) & ~filters.COMMAND, on_message))
    print(f"qwen-tg-bridge up | model={MODEL} base={BASE_URL} allow={sorted(ALLOWED)} "
          f"idle_timeout={IDLE_TIMEOUT}s max_timeout={MAX_TIMEOUT}s workdir={WORKDIR}")
    app.run_polling()


if __name__ == "__main__":
    main()
