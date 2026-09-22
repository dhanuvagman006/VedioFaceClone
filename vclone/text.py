"""Text clean-up, English normalisation and sentence-aware chunking."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ----------------------------------------------------------------------------- clean-up

_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "…": "...", "—": ", ", "―": ", ", "–": " - ", "−": "-",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u200b": "", "\ufeff": "",
})
_KEEP_SYMBOLS = set("%$€£₹¥&+=@#°/")


def clean_text(text: str) -> str:
    """Remove emoji, markdown decoration and odd whitespace; keep line structure."""
    text = unicodedata.normalize("NFC", text).translate(_CHAR_MAP)
    kept = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t":
            kept.append(ch)
        elif cat.startswith("C") or "\ufe00" <= ch <= "\ufe0f":
            continue  # control / format chars, variation selectors
        elif cat in ("So", "Sk") and ch not in _KEEP_SYMBOLS:
            continue  # emoji and pictographs
        else:
            kept.append(ch)
    text = "".join(kept).replace("\r", "")
    text = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]*", "", text, flags=re.M)      # markdown headings
    text = re.sub(r"^[ \t]*(?:[-*+•]|\d+[.)])[ \t]+", "", text, flags=re.M)  # list bullets
    text = re.sub(r"[*`~]+", "", text).replace("_", " ")
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(?:[ \t]*[,;:])+", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# ----------------------------------------------------------------------------- English

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e",
    "fig", "inc", "ltd", "co", "corp", "dept", "approx", "est", "vol", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k", "a.m", "p.m",
}
_WORD_ABBREV = [
    (r"\bMrs\.", "Missus"), (r"\bMr\.", "Mister"), (r"\bMs\.", "Miz"), (r"\bDr\.", "Doctor"),
    (r"\bProf\.", "Professor"), (r"\bvs\.?(?=\s)", "versus"), (r"\betc\.", "et cetera"),
    (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"),
    (r"\bJan\.", "January"), (r"\bFeb\.", "February"), (r"\bMar\.", "March"), (r"\bApr\.", "April"),
    (r"\bJun\.", "June"), (r"\bJul\.", "July"), (r"\bAug\.", "August"), (r"\bSept?\.", "September"),
    (r"\bOct\.", "October"), (r"\bNov\.", "November"), (r"\bDec\.", "December"),
]
_CURRENCY = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
    "₹": ("rupee", "rupees", "paisa", "paise"),
    "¥": ("yen", "yen", "", ""),
}
_SCALES = {"k": "thousand", "thousand": "thousand", "m": "million", "mn": "million",
           "million": "million", "b": "billion", "bn": "billion", "billion": "billion",
           "trillion": "trillion"}


def _words(n) -> str:
    from num2words import num2words
    return num2words(n).replace(",", "")


def _money(m: re.Match) -> str:
    one, many, sub_one, sub_many = _CURRENCY[m.group(1)]
    number, scale = m.group(2).replace(",", ""), (m.group(3) or "").strip().lower()
    if scale:
        return f"{_words(float(number) if '.' in number else int(number))} {_SCALES[scale]} {many}"
    whole, _, frac = number.partition(".")
    units, cents = int(whole or 0), int((frac + "00")[:2]) if frac else 0
    parts = []
    if units or not cents:
        parts.append(f"{_words(units)} {one if units == 1 else many}")
    if cents and sub_one:
        parts.append(f"{_words(cents)} {sub_one if cents == 1 else sub_many}")
    return " and ".join(parts)


def _time(m: re.Match) -> str:
    hour, minute, suffix = int(m.group(1)), int(m.group(2)), (m.group(3) or "")
    if hour > 24 or minute > 59:
        return m.group(0)
    spoken = _words(hour)
    if minute == 0:
        spoken += "" if suffix else " o'clock"
    elif minute < 10:
        spoken += f" oh {_words(minute)}"
    else:
        spoken += f" {_words(minute)}"
    if suffix:
        spoken += " " + " ".join(suffix.replace(".", "").strip().upper())
    return spoken


def _number(m: re.Match) -> str:
    raw = m.group(0)
    digits = raw.replace(",", "")
    if "." in digits:
        whole, frac = digits.split(".", 1)
        return f"{_words(int(whole))} point {' '.join(_words(int(d)) for d in frac)}"
    value = int(digits)
    if "," not in raw and len(digits) == 4 and 1100 <= value <= 2099:
        from num2words import num2words
        return num2words(value, to="year")
    if len(digits) > 15:  # phone numbers / ids: read digit by digit
        return " ".join(_words(int(d)) for d in digits)
    return _words(value)


def _decade(m: re.Match) -> str:
    digits = m.group(1)
    if not digits.endswith("0"):
        return m.group(0)
    from num2words import num2words
    words = num2words(int(digits), to="year") if len(digits) == 4 else _words(int(digits))
    return words[:-1] + "ies" if words.endswith("y") else words + "s"


def normalize_english(text: str) -> str:
    """Spell out numbers, currency, times and common abbreviations (English only)."""
    from num2words import num2words

    for pattern, repl in _WORD_ABBREV:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"([$£€₹¥])\s?(\d[\d,]*(?:\.\d+)?)(\s?(?:thousand|million|billion|trillion|bn|mn|[kKmMbB])\b)?",
                  _money, text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b(\s?[aApP]\.?[mM]\.?)?", _time, text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: f"{m.group(1)} percent", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?°\s?([CF])\b",
                  lambda m: f"{m.group(1)} degrees {'Celsius' if m.group(2) == 'C' else 'Fahrenheit'}", text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), text)
    text = re.sub(r"(?<![\d'’])['’]?(\d{2}|1[1-9]\d{2}|20\d{2})s\b", _decade, text)   # 1990s, '90s
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)             # F5, MP3, 4K
    text = re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])|(?<![\d.])\d+(?:\.\d+)?(?![\d])", _number, text)
    text = text.replace("&", " and ").replace("°", " degrees").replace("@", " at ")
    text = re.sub(r"(?<=\s)\+(?=\s)", "plus", text)
    return re.sub(r"[ \t]+", " ", text).strip()


# ----------------------------------------------------------------------------- chunking

@dataclass
class Chunk:
    text: str           # as written (used to check the result)
    pause_after: float  # seconds of silence inserted after this chunk
    speak: str = ""     # what the engine reads (numbers spelled out for F5)

    def __post_init__(self):
        self.speak = self.speak or self.text


_TERMINATOR = re.compile(r"""([.!?]+|…)(["'”’)\]]*)\s+""")
_CJK_SPLIT = re.compile(r"(?<=[。！？；])")
_SENTENCE_END = tuple(".!?…。！？\"'”’)")


def split_sentences(text: str) -> list[str]:
    sentences, start = [], 0
    for m in _TERMINATOR.finditer(text):
        if m.group(1) == ".":
            prev = re.search(r"([A-Za-z][A-Za-z.]*)$", text[start:m.start()])
            if prev:
                word = prev.group(1).rstrip(".")
                if word.lower() in _ABBREV or (len(word) == 1 and word.isupper()):
                    continue  # "Dr. Smith", "J. K. Rowling"
        sentences.append(text[start:m.end(2)].strip())
        start = m.end()
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return [p.strip() for s in sentences for p in _CJK_SPLIT.split(s) if p.strip()]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Break an over-long sentence at clause punctuation, then at spaces."""
    if len(sentence) <= max_chars:
        return [sentence]
    out, cur = [], ""
    for part in re.split(r"(?<=[,;:])\s+|(?<=[，；：])\s*", sentence):
        words = part.split() if len(part) > max_chars else [part]
        for word in words:
            while len(word) > max_chars:  # unspaced scripts
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(word[:max_chars])
                word = word[max_chars:]
            if cur and len(cur) + 1 + len(word) > max_chars:
                out.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return [p for p in out if p]


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    pieces = [p for s in sentences for p in _split_long(s, max_chars)]
    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + 1 + len(piece) > max_chars:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    # Tiny fragments sound odd on their own; fold each into its shorter neighbour.
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 30:
            near = [j for j in (i - 1, i + 1)
                    if 0 <= j < len(chunks) and len(chunks[i]) + len(chunks[j]) < max_chars * 1.3]
            if near:
                a, b = sorted((i, min(near, key=lambda j: len(chunks[j]))))
                chunks[a:b + 1] = [f"{chunks[a]} {chunks[b]}"]
                i = a
                continue
        i += 1
    return chunks


def _paragraphs(text: str) -> list[str]:
    """Blank lines separate paragraphs. A single line break continues the sentence
    when the next line starts in lower case (hard-wrapped prose); otherwise it ends it."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        merged = lines[0]
        for line in lines[1:]:
            if line[0].islower() or merged.endswith((",", ";", ":", "-")):
                merged += " " + line
            else:
                merged += (" " if merged.endswith(_SENTENCE_END) else ". ") + line
        paragraphs.append(merged)
    return paragraphs


def plan_chunks(text: str, max_chars: int, sentence_pause: float = 0.3,
                paragraph_pause: float = 0.7, clause_pause: float = 0.12) -> list[Chunk]:
    chunks: list[Chunk] = []
    for paragraph in _paragraphs(text):
        pieces = _pack(split_sentences(paragraph), max_chars)
        for i, piece in enumerate(pieces):
            if i == len(pieces) - 1:
                pause = paragraph_pause
            elif piece.endswith(_SENTENCE_END):
                pause = sentence_pause
            else:
                pause = clause_pause
            chunks.append(Chunk(piece, pause))
    if chunks:
        chunks[-1].pause_after = 0.0
    return chunks
