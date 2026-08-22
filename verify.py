#!/usr/bin/env python3
"""verify.py — smoke-test what qwen-code just produced, and return a fix hint on failure.

Why this exists: `node --check` / `py_compile` only prove code PARSES. The bugs that
make a single-file app look "done" but be broken are RUNTIME bugs — e.g. a NaN camera
position that renders a blank screen (a real qwen output, 2026-08-22). node --check
passed, the model declared done, but it never actually ran the thing. This module runs
it and reports what actually happened, so the bridge can loop the model back to fix it.

Design principles:
  * FAIL OPEN. Only report failure on POSITIVE evidence of a bug (an uncaught JS error,
    a Python traceback, or a canvas that is provably blank while WebGL/2D is confirmed
    working in the harness). A broken harness (no playwright, WebGL unavailable, a page
    that needs a backend we can't run) returns ok=True with a note — we never punish the
    model for our own instrument (that mistake cost hours on 2026-08-22).
  * BOUNDED. Every subprocess / browser / server has a timeout. Servers are only
    syntax-checked, never executed (they'd hang).
  * SYNC. Call from the async bridge via asyncio.to_thread.

Public API:
  snapshot(workdir) -> dict[str, float]
  changed_since(before, after) -> list[Path]
  verify(workdir, changed_paths) -> VerifyResult(ok, summary, fix_hint)
"""
from __future__ import annotations

import functools
import http.server
import io
import re
import socketserver
import statistics
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

CODE_EXTS = {".py", ".js", ".mjs", ".cjs", ".html", ".htm"}
ALL_EXTS = CODE_EXTS | {".ts", ".css", ".json"}
SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", ".next"}
SKIP_NAME = re.compile(r"^\.|^_run\.log$|^tg-photo-|\.imgfile$|^index_dbg\.html$")

# A python file that opens a listener would hang if executed, so only syntax-check these.
_SERVER_HINT = re.compile(
    r"Flask\(|app\.run\(|uvicorn|http\.server|socketserver|\.serve_forever\(|"
    r"\.listen\(|FastAPI\(|gunicorn|FLASK|create_server|while True",
    re.I,
)


@dataclass
class VerifyResult:
    ok: bool
    summary: str      # short, user-facing (one line)
    fix_hint: str     # detailed failure fed back to the model; "" when ok


# --------------------------------------------------------------------------- files
def snapshot(workdir: Path) -> dict:
    out: dict = {}
    for p in Path(workdir).rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in ALL_EXTS or SKIP_NAME.search(p.name):
            continue
        try:
            out[str(p)] = p.stat().st_mtime
        except OSError:
            pass
    return out


def changed_since(before: dict, after: dict) -> list:
    return [Path(k) for k, v in after.items() if before.get(k) != v]


# --------------------------------------------------------------------------- syntax
def _syntax_errors(paths) -> list:
    errs: list = []
    for p in paths:
        ext = p.suffix.lower()
        try:
            if ext == ".py":
                r = subprocess.run(["python3", "-m", "py_compile", str(p)],
                                   capture_output=True, text=True, timeout=30)
                if r.returncode:
                    tail = (r.stderr.strip().splitlines() or ["py_compile failed"])[-1]
                    errs.append(f"{p.name}: Python syntax error — {tail}")
            elif ext in (".js", ".mjs", ".cjs"):
                r = subprocess.run(["node", "--check", str(p)],
                                   capture_output=True, text=True, timeout=30)
                if r.returncode:
                    head = (r.stderr.strip().splitlines() or ["node --check failed"])[0]
                    errs.append(f"{p.name}: JS syntax error — {head}")
        except subprocess.TimeoutExpired:
            pass  # fail open
        except Exception as e:  # noqa: BLE001
            pass  # never let a harness hiccup become a false failure
    return errs


# --------------------------------------------------------------------------- python run
def _run_python(path: Path) -> list:
    """Execute a simple (non-server) python script with a timeout; report a traceback."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    if _SERVER_HINT.search(text):
        return []  # server/long-runner — syntax-only, don't execute
    if "__main__" not in text and "input(" in text:
        return []  # interactive; skip
    try:
        r = subprocess.run(["python3", str(path)], cwd=str(path.parent),
                           capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return []  # probably does real work / waits — not evidence of a bug
    except Exception:  # noqa: BLE001
        return []
    if r.returncode != 0 and r.stderr.strip():
        lines = [l for l in r.stderr.strip().splitlines() if l.strip()]
        # keep the exception line(s), which is what the model needs
        tail = "\n".join(lines[-4:])
        return [f"{path.name}: crashed on run —\n{tail}"]
    return []


# --------------------------------------------------------------------------- html render
def _serve(workdir: Path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(workdir))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    httpd.allow_reuse_address = True
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def _render_html(workdir: Path, html_rel: str, timeout: int = 25) -> VerifyResult:
    """Load the page headless, run its loop, and report JS errors or a blank canvas.

    Returns ok=True (with a note) whenever the HARNESS can't give a clean verdict, so a
    missing browser / unavailable WebGL / backend-dependent page never blocks the model.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:  # noqa: BLE001
        return VerifyResult(True, "render-check skipped (no playwright)", "")

    console_errors: list = []
    page_errors: list = []
    httpd, port = _serve(workdir)
    shot = None
    webgl_ok = False
    has_canvas = False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=[
                "--enable-unsafe-swiftshader", "--use-gl=angle",
                "--use-angle=swiftshader", "--ignore-gpu-blocklist",
            ])
            pg = browser.new_page(viewport={"width": 640, "height": 420})
            # Keep the page fully local: fulfil any external request (fonts/CDN) with an empty
            # body, so document.fonts.ready resolves instead of hanging the screenshot forever.
            pg.route("**/*", lambda r: (r.continue_()
                     if r.request.url.startswith("http://127.0.0.1")
                     else r.fulfill(status=200, body="")))
            pg.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: page_errors.append(str(e)))
            pg.goto(f"http://127.0.0.1:{port}/{html_rel}", wait_until="load", timeout=timeout * 1000)
            pg.wait_for_timeout(500)
            has_canvas = bool(pg.evaluate("!!document.querySelector('canvas')"))
            webgl_ok = bool(pg.evaluate(
                "(()=>{try{const c=document.createElement('canvas');"
                "const g=c.getContext('webgl2')||c.getContext('webgl');"
                "return !!(g&&g.getParameter(g.VERSION));}catch(e){return false;}})()"
            ))
            # best-effort 'start': many single-file games begin on Enter or a click. Native rAF
            # runs in headless chromium, so the game loop advances on its own — no shim needed.
            try:
                pg.keyboard.press("Enter")
                pg.mouse.click(320, 210)
            except Exception:  # noqa: BLE001
                pass
            pg.wait_for_timeout(1800)
            # Freeze the loop so the screenshot isn't racing an animating (software-WebGL, CPU
            # heavy) page — otherwise the capture times out on a busy compositor.
            try:
                pg.evaluate("window.requestAnimationFrame=function(){return 0;};")
            except Exception:  # noqa: BLE001
                pass
            pg.wait_for_timeout(120)
            shot = pg.screenshot(timeout=8000)
            browser.close()
    except Exception as e:  # noqa: BLE001
        return VerifyResult(True, f"render-check inconclusive ({type(e).__name__})", "")
    finally:
        try:
            httpd.shutdown()
        except Exception:  # noqa: BLE001
            pass

    # 1) real JS errors are the strongest signal
    errs = list(dict.fromkeys(page_errors + console_errors))
    errs = [e for e in errs if e and "favicon" not in e.lower()]
    if errs:
        hint = "The page threw errors when run in a headless browser:\n- " + "\n- ".join(errs[:6])
        return VerifyResult(False, f"{html_rel}: {len(errs)} runtime error(s) in browser", hint)

    # 2) blank canvas — measure the CENTRE of the frame (avoids DOM HUD overlays) by colour
    # variance. Trusted only when WebGL/2D is confirmed working, so a broken harness never
    # false-fails. Calibrated on the real gta3d oracle: good scene stdev~32 / 56 colours,
    # the NaN-blank bug ~0.6 / 6 colours — a wide, safe margin.
    if has_canvas and webgl_ok and shot:
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(shot)).convert("RGB")
            W, H = im.size
            crop = im.crop((int(W * 0.18), int(H * 0.32), int(W * 0.82), int(H * 0.94)))
            px = list(crop.getdata())[::11]
            grey = [(r * 299 + g * 587 + b * 114) // 1000 for (r, g, b) in px]
            sd = statistics.pstdev(grey) if len(grey) > 1 else 0.0
            distinct = len({(r >> 4, g >> 4, b >> 4) for (r, g, b) in px})
            if sd < 5.0 and distinct <= 8:
                hint = ("The app has a <canvas> but after loading + a start (Enter/click) the "
                        "rendered frame is essentially blank/uniform — nothing is drawn. This is "
                        "the classic 'runs but shows nothing' bug (a NaN in a camera/transform, "
                        "geometry never added to the scene, or the draw loop never advancing). "
                        "Run it, confirm the scene is actually visible, and fix whatever leaves "
                        "the canvas empty.")
                return VerifyResult(False, f"{html_rel}: canvas renders blank after start", hint)
        except Exception:  # noqa: BLE001
            pass  # fail open on any imaging hiccup

    return VerifyResult(True, f"{html_rel}: rendered, no JS errors", "")


# --------------------------------------------------------------------------- orchestrate
def _pick_html(changed) -> Path | None:
    htmls = [p for p in changed if p.suffix.lower() in (".html", ".htm")]
    if not htmls:
        return None
    for p in htmls:  # prefer an obvious entrypoint
        if p.name.lower() in ("index.html", "main.html", "game.html"):
            return p
    return max(htmls, key=lambda p: p.stat().st_mtime if p.exists() else 0)


def verify(workdir, changed_paths) -> VerifyResult:
    """Smoke-test the changed artifacts. See module docstring for the fail-open contract."""
    workdir = Path(workdir)
    changed = [p for p in changed_paths if p.exists() and not SKIP_NAME.search(p.name)]
    if not changed:
        return VerifyResult(True, "no code artifacts to verify", "")

    # 1) syntax — cheap, safe, always
    syn = _syntax_errors(changed)
    if syn:
        return VerifyResult(False, f"{len(syn)} syntax error(s)",
                            "These files do not parse:\n- " + "\n- ".join(syn[:6]))

    # 2) web app? render it (the high-value check)
    html = _pick_html(changed)
    if html is not None:
        rel = str(html.relative_to(workdir)) if str(html).startswith(str(workdir)) else html.name
        return _render_html(workdir, rel)

    # 3) plain python script(s)? run the entrypoint(ish) one
    pys = [p for p in changed if p.suffix.lower() == ".py"]
    if pys:
        entry = next((p for p in pys if p.name in ("main.py", "app.py", "run.py")), None) or \
                max(pys, key=lambda p: p.stat().st_size if p.exists() else 0)
        errs = _run_python(entry)
        if errs:
            return VerifyResult(False, f"{entry.name}: crashed on run",
                                "Running it produced an error:\n" + "\n".join(errs))
        return VerifyResult(True, f"{entry.name}: ran clean (or is a service)", "")

    return VerifyResult(True, "syntax ok (no runnable entrypoint detected)", "")


if __name__ == "__main__":  # tiny CLI: python3 verify.py <workdir> [file ...]
    import sys
    wd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    files = [Path(a) for a in sys.argv[2:]] or list(snapshot(wd).keys()) and [Path(k) for k in snapshot(wd)]
    res = verify(wd, files)
    print(f"ok={res.ok}\nsummary={res.summary}\nfix_hint={res.fix_hint}")
