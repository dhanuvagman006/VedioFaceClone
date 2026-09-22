"""Command line: speak.bat (Windows) or ./speak.sh (Linux, Colab) <recording> "text to read" [options]"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .models import COMMAND

EXAMPLES = r"""
examples:
  speak.bat person.mp4 -f script.txt -o talking.mp4     (30 s video in -> lip-synced video out)
  speak.bat my_voice.m4a "Hello! This is my cloned voice."
  speak.bat my_voice.wav --text-file story.txt -o story.mp3
  speak.bat my_voice.wav "Quick test" --quality fast
  speak.bat my_voice.wav "Bonjour tout le monde" --language french
  speak.bat my_voice.wav "Some text" --engine f5 --quality max

tips:
  * Use 10-30 s of clean, natural speech (quiet room, no music). The tool picks the best
    5-12 s by itself, transcribes it once and caches it in voices\.
  * If a word in the reference transcript is wrong, fix voices\<name>\ref.txt and re-run.
""".replace("speak.bat", COMMAND).replace("\\", os.sep)


SCRIPT_HELP = """Record yourself reading this aloud (about 10 seconds):

    {script}

  * Quiet room, no fan/TV/music; hold the mic or phone about 20 cm from your mouth.
  * Speak naturally, the way you want the clone to sound. Don't rush.
  * Windows Sound Recorder or your phone's voice memo app is fine.

Then use that recording, e.g.:  {command} "{example}" "Any text you like"
The tool recognizes the script and uses its exact words, so the clone isn't thrown off by a bad transcript.
"""
SCRIPT_EXAMPLE = r"C:\path\to\Recording.m4a" if os.name == "nt" else "path/to/recording.m4a"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speak", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EXAMPLES,
        description="Read any text aloud in your own cloned voice, on your local NVIDIA GPU.")
    p.add_argument("voice", nargs="?", help="recording of the voice (wav, mp3, m4a, ...) or a video of the person "
                                            "(mp4, mov, ...): a video gives a lip-synced talking video")
    p.add_argument("words", nargs="?", metavar="text", help="the text to read (or use --text-file)")
    p.add_argument("-v", "--voice", dest="voice_opt", metavar="FILE", help="same as the first argument")
    p.add_argument("-t", "--text", dest="text_opt", metavar="TEXT", help="text to read ('-' reads stdin)")
    p.add_argument("-f", "--text-file", metavar="FILE", help="read the text from a .txt file")
    p.add_argument("-o", "--out", metavar="FILE", help="output file (.wav .flac .mp3 .m4a .ogg); "
                                                        "default: outputs\\<time>_<words>.wav")
    p.add_argument("--script", action="store_true",
                   help="print a short text to read aloud for your voice recording, then exit")

    q = p.add_argument_group("quality")
    q.add_argument("-q", "--quality", choices=["fast", "high", "max"], default="high",
                   help="fast = 1 take; high = best of 3 takes (default); max = best of 5, finer F5 sampling")
    q.add_argument("--engine", choices=["qwen", "f5"], default="qwen",
                   help="qwen = Qwen3-TTS (default, 10 languages); f5 = F5-TTS (English/Chinese)")
    q.add_argument("--takes", type=int, help="override how many takes to generate per chunk")
    q.add_argument("--retries", type=int, default=1, help="extra rounds for parts Whisper found misread (default 1)")
    q.add_argument("--check", action="store_true", help="verify with Whisper even with a single take")
    q.add_argument("--seed", type=int, help="random seed, for reproducible output")

    v = p.add_argument_group("voice & language")
    v.add_argument("-l", "--language", default="auto",
                   help="language of the text: auto (default), english, chinese, japanese, korean, german, "
                        "french, russian, portuguese, spanish, italian")
    v.add_argument("--ref-text", help="exact words spoken in your recording (a recording under 15 s is used "
                                      "whole; a longer one is matched against these words)")
    v.add_argument("--ref-language", help="language spoken in your recording, e.g. english or hindi "
                                          "(default: detected)")
    v.add_argument("--refresh-voice", action="store_true", help="re-process the recording instead of using the cache")
    v.add_argument("--xvector-only", action="store_true",
                   help="qwen: clone from the speaker embedding only (use if your recording is in an "
                        "unsupported language)")
    v.add_argument("--qwen-size", choices=["0.6b", "1.7b"], default="0.6b",
                   help="qwen model size; 1.7b needs about 6 GB of VRAM (default 0.6b)")
    v.add_argument("--temperature", type=float, help="qwen: sampling temperature (default 0.9)")
    v.add_argument("--top-p", type=float, help="qwen: nucleus sampling (default 1.0)")
    v.add_argument("--steps", type=int, help="f5: flow-matching steps (32 default, 64 for max)")
    v.add_argument("--speed", type=float, default=1.0, help="f5: speaking rate multiplier (default 1.0)")

    t = p.add_argument_group("talking video")
    t.add_argument("--face", metavar="FILE",
                   help="video (or photo) of the person to lip-sync; default: your recording when it is a video")
    t.add_argument("--refresh-face", action="store_true", help="re-analyse the face video instead of using the cache")

    o = p.add_argument_group("output")
    o.add_argument("--pause", type=float, default=0.3, help="silence between sentences in seconds (0.3)")
    o.add_argument("--paragraph-pause", type=float, default=0.7, help="silence between paragraphs (0.7)")
    o.add_argument("--loudness", type=float, default=-18.0, help="target loudness in LUFS (-18)")
    o.add_argument("--sample-rate", type=int, help="resample output, e.g. 44100 or 48000 (native 24000)")
    o.add_argument("--save-takes", action="store_true", help="also save every take next to the output")
    o.add_argument("-V", "--verbose", action="store_true", help="show the score of every take")
    o.add_argument("--device", help="torch device, e.g. cuda:0 or cpu (auto)")
    return p


def _read_text(args) -> str:
    if args.text_file:
        raw = Path(args.text_file).read_bytes()
        for enc in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")
    text = args.text_opt if args.text_opt is not None else args.words
    if text == "-":
        return sys.stdin.read()
    return text or ""


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.script:
        from .reference import ENROLL_SCRIPT
        print(SCRIPT_HELP.format(script=ENROLL_SCRIPT, command=COMMAND, example=SCRIPT_EXAMPLE))
        return 0
    args.ref = args.voice_opt or args.voice
    args.text = _read_text(args)
    if not args.ref or not args.text.strip():
        parser.print_usage()
        print(f'speak: error: give your voice recording and the text, e.g.  {COMMAND} me.wav "Hello there"')
        return 2

    from .models import setup_env
    setup_env()
    from .pipeline import run
    try:
        run(args, log=lambda msg: print(msg, flush=True))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.stdout.flush()
        if "out of memory" in str(exc).lower():
            print("error: the GPU ran out of memory. Close other GPU apps, use --quality fast, "
                  "or split the text.", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
