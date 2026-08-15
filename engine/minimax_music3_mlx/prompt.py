from __future__ import annotations

import re

SPECIAL_TOKEN_IDS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|audio_cfg|>": 151654,
    "<|audio_start|>": 151669,
    "<|audio_end|>": 151670,
    "<|caption_start|>": 151671,
    "<|caption_end|>": 151672,
    "<|lyrics_start|>": 151673,
    "<|lyrics_end|>": 151674,
}
AUDIO_CODE_OFFSET = 151675


def clean_caption(caption: str) -> str:
    def replace_special(match: re.Match[str]) -> str:
        parts = match.group(1).strip().split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else match.group(1).strip()

    text = re.sub(r"<\|([^|]*)\|>", replace_special, caption)
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", raw_line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    lines = []
    for line in lyrics.splitlines():
        match = re.match(r"^\s*((?:\[[^\]]+\][ \t]*)+)", line)
        lines.append(match.group(1).strip() if match else line)
    text = "\n".join(lines).replace("] ", "]\n").replace(" [", "\n[").replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda m: f"[{m.group(1).lower()}]", text)
    return f"[start]\n{text}"


def build_prompt(caption: str, lyrics: str) -> str:
    return (
        "<|im_start|><|caption_start|>"
        + clean_caption(caption)
        + "<|caption_end|><|lyrics_start|>"
        + normalize_lyrics(lyrics)
        + "<|lyrics_end|><|im_end|><|audio_start|>"
    )
