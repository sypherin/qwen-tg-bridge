#!/usr/bin/env python3
"""hear — give any agent ears via local faster-whisper (CPU, int8).

Usage:
  hear /path/to/audio.ogg             # transcribe, print text to stdout
  hear /path/to/file.mp3 --lang en    # force language (default: auto-detect)

Exit 0 = transcription ran (stdout may be empty if no intelligible speech);
exit 1 = failure (bad file, model missing); exit 2 = usage.
Used by: qwen-tg-bridge (voice notes). Mirrors ~/bin/see for audio.
Model via HEAR_MODEL env (default "small", cached in ~/.cache/huggingface).
"""
import os
import sys

from faster_whisper import WhisperModel

MODEL_NAME = os.environ.get("HEAR_MODEL", "small")

_lang = None
_args = sys.argv[1:]
if "--lang" in _args:
    i = _args.index("--lang")
    if i + 1 < len(_args):
        _lang = _args[i + 1]
        del _args[i:i + 2]
    else:
        del _args[i]
paths = [a for a in _args if not a.startswith("-")]
if not paths:
    print(__doc__.strip(), file=sys.stderr)
    sys.exit(2)


def main() -> int:
    try:
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(
            paths[0], beam_size=5, vad_filter=True, language=_lang,
        )
        text = "".join(s.text for s in segments).strip()
    except Exception as e:  # noqa: BLE001 — CLI: surface any failure as exit 1
        print(f"hear: {e}", file=sys.stderr)
        return 1
    print(text)
    return 0


sys.exit(main())
