"""Tests for verify.py (the run-and-fix smoke-tester) and the bridge fix-loop flow.

Run: python3 tests/test_verify.py    (no pytest needed)
The heavy headless-render path is covered separately by the gta3d good/bug oracle; these
tests cover the pure logic + the loop control flow (with a stubbed model runner).
"""
import asyncio
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import verify  # noqa: E402


def _tmp() -> pathlib.Path:
    return pathlib.Path(tempfile.mkdtemp(prefix="vtest-"))


def test_snapshot_and_changed():
    d = _tmp()
    (d / "a.py").write_text("x = 1\n")
    before = verify.snapshot(d)
    assert str(d / "a.py") in before
    time.sleep(0.01)
    (d / "a.py").write_text("x = 2\n")
    (d / "b.js").write_text("var y = 1;\n")
    (d / "_run.log").write_text("noise\n")  # must be ignored
    after = verify.snapshot(d)
    names = {p.name for p in verify.changed_since(before, after)}
    assert "a.py" in names and "b.js" in names
    assert "_run.log" not in names
    print("ok  snapshot / changed_since (skips _run.log)")


def test_syntax_errors():
    d = _tmp()
    good = d / "g.py"; good.write_text("def f():\n    return 1\n")
    badpy = d / "b.py"; badpy.write_text("def f(:\n    pass\n")
    badjs = d / "b.js"; badjs.write_text("function ( {\n")
    errs = verify._syntax_errors([good, badpy, badjs])
    assert not any("g.py" in e for e in errs)
    assert any("b.py" in e for e in errs)
    assert any("b.js" in e for e in errs)
    print("ok  _syntax_errors (py + js, good file clean)")


def test_python_run_and_server_skip():
    d = _tmp()
    crash = d / "c.py"; crash.write_text("raise ValueError('boom-xyz')\n")
    clean = d / "ok.py"; clean.write_text("print('hi')\n")
    server = d / "s.py"; server.write_text("import http.server\nwhile True:\n    pass\n")
    r = verify._run_python(crash)
    assert r and "boom-xyz" in r[0]
    assert verify._run_python(clean) == []
    assert verify._run_python(server) == []  # server -> not executed
    print("ok  _run_python (crash caught, clean ok, server skipped)")


def test_verify_python_crash_and_noop():
    d = _tmp()
    f = d / "main.py"; f.write_text("import sys\nsys.exit('bad thing happened')\n")
    r = verify.verify(d, [f])
    assert r.ok is False and "crash" in r.summary.lower()
    assert verify.verify(d, []).ok is True  # nothing to verify
    print("ok  verify() python crash -> fail, no-artifacts -> ok")


def test_fix_loop_flow():
    """Stub the model runner: round 1 writes a broken file, round 2 fixes it.
    Proves run_qwen_verified catches the failure and loops to a pass."""
    import bridge
    d = _tmp()
    bridge.WORKDIR = d
    bridge.VERIFY_ON = True
    bridge.MAX_FIX_ROUNDS = 2
    state = {"n": 0}

    async def fake_run(prompt, notify=None):
        state["n"] += 1
        if state["n"] == 1:
            (d / "main.py").write_text("raise RuntimeError('first attempt broken')\n")
            return "built it"
        (d / "main.py").write_text("print('works now')\n")
        return "fixed it"

    bridge.run_qwen = fake_run
    msgs = []

    async def notify(m):
        msgs.append(m)

    out = asyncio.run(bridge.run_qwen_verified("build main.py", notify=notify))
    assert "auto-verified" in out, out
    assert state["n"] == 2, f"expected 1 fix round, got {state['n']} runs"
    assert any("fixing" in m for m in msgs), msgs
    print("ok  fix-loop flow (caught on round 1, fixed on round 2)")


def test_fix_loop_gives_up_after_cap():
    """If the model never fixes it, the loop stops at the cap and returns a warning."""
    import bridge
    d = _tmp()
    bridge.WORKDIR = d
    bridge.VERIFY_ON = True
    bridge.MAX_FIX_ROUNDS = 2
    runs = {"n": 0}

    async def always_broken(prompt, notify=None):
        runs["n"] += 1
        (d / "main.py").write_text("raise RuntimeError('still broken')\n")
        return "attempt"

    bridge.run_qwen = always_broken
    out = asyncio.run(bridge.run_qwen_verified("build main.py", notify=None))
    assert "still failing" in out, out
    assert runs["n"] == 3, runs  # initial + 2 fix rounds
    print("ok  fix-loop gives up after MAX_FIX_ROUNDS (returns reply anyway)")


if __name__ == "__main__":
    test_snapshot_and_changed()
    test_syntax_errors()
    test_python_run_and_server_skip()
    test_verify_python_crash_and_noop()
    test_fix_loop_flow()
    test_fix_loop_gives_up_after_cap()
    print("\nALL PASS")
