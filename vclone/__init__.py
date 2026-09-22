"""Local voice cloning: read any text in your own voice on your NVIDIA GPU.

    from vclone import speak
    speak("my_voice.m4a", "Hello there!", "hello.wav", quality="fast")
"""
from __future__ import annotations


def speak(voice, text: str, out=None, **options):
    """Python API. `options` are the long CLI flags with '_' for '-', e.g. engine="f5", takes=5,
    language="french", ref_text="...", sample_rate=48000. Returns the output path."""
    from .cli import build_parser
    from .models import setup_env
    from .pipeline import run

    setup_env()
    args = build_parser().parse_args([])
    args.ref, args.text, args.out = str(voice), text, out
    for key, value in options.items():
        if not hasattr(args, key):
            raise TypeError(f"unknown option '{key}'")
        setattr(args, key, value)
    return run(args)
