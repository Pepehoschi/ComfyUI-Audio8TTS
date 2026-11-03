"""Parse tagged script text into segments bound to voice-set references.

Grammar (kept deliberately minimal, mirroring the JS extension rules):
  [tag]      switches the active voice reference from this position until
             the next tag; exact match against the voice set (case
             sensitive); a tag mid-sentence is allowed but warned.
  [[  ]]     escapes producing literal square brackets in the text.
  [unknown]  kept verbatim in the text and reported in the log (never
             silently dropped), so prose like "[almost]" survives intact.
Adjacent pieces of the same tag merge into one segment; the leading
un-tagged span (if any) renders with the TTS default reference.
"""

import re

from .audio8_chunking import SENTENCE_ENDINGS, split_text, TextChunk
from .audio8_types import Audio8Script, ScriptSegment, VoiceSet

TOKEN_PATTERN = re.compile(r"\[\[|\]\]|\[([^\[\]\n]{1,64})\]")
MIN_SEGMENT_CHARS = 10


def plan_script_chunks(script, chunking="auto", target_chunk_chars=130,
                       max_chunk_chars=150, sentence_pause_ms=180,
                       paragraph_pause_ms=350):
    """Map script segments onto (text, reference, pause_after_ms) chunks.

    Tag boundaries act like paragraph boundaries. Pure function: no torch
    state, no model - directly unit-testable.
    """

    if not script or not script.segments:
        return []
    planned = []
    segments = script.segments
    for index, segment in enumerate(segments):
        if chunking == "disabled":
            chunks = [TextChunk(segment.text.strip(), 0)]
        else:
            chunks = split_text(segment.text, target_chunk_chars, max_chunk_chars,
                                sentence_pause_ms, paragraph_pause_ms)
        for chunk_index, chunk in enumerate(chunks):
            is_last_chunk = chunk_index == len(chunks) - 1
            if is_last_chunk:
                pause = 0 if index == len(segments) - 1 else paragraph_pause_ms
            else:
                pause = chunk.pause_after_ms
            planned.append((chunk.text, segment.reference, pause))
    return [entry for entry in planned if entry[0]]


def _is_clean_boundary(text_before_tag: str) -> bool:
    stripped = text_before_tag.rstrip()
    if not stripped:
        return True
    return stripped[-1] in SENTENCE_ENDINGS


def parse_script(text: str, voice_set: VoiceSet) -> tuple[Audio8Script, list[str]]:
    """Return (script, log_lines); unknown tags warn but never raise."""

    if not isinstance(text, str):
        raise TypeError("Script text must be a string")
    if not isinstance(voice_set, VoiceSet):
        raise TypeError("parse_script needs a VoiceSet")

    warnings: list[str] = []
    segments: list[ScriptSegment] = []
    current_tag: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_tag
        piece = "".join(buffer).strip()
        buffer.clear()
        if not piece:
            return
        if len(piece) < MIN_SEGMENT_CHARS and current_tag is not None:
            warnings.append(f"Segment for tag '{current_tag}' is very short "
                            f"({len(piece)} chars): {piece!r}")
        if segments and segments[-1].tag == current_tag:
            merged = f"{segments[-1].text} {piece}"
            segments[-1] = ScriptSegment(current_tag,
                                         voice_set.get(current_tag) if current_tag else None,
                                         merged)
        else:
            segments.append(ScriptSegment(current_tag,
                                          voice_set.get(current_tag) if current_tag else None,
                                          piece))

    position = 0
    for match in TOKEN_PATTERN.finditer(text):
        buffer.append(text[position:match.start()])
        position = match.end()
        token = match.group(0)
        if token == "[[":
            buffer.append("[")
            continue
        if token == "]]":
            buffer.append("]")
            continue
        tag = match.group(1)
        reference = voice_set.get(tag)
        if reference is None:
            if voice_set.entries:
                warnings.append(f"Unknown tag '[{tag}]' kept as literal text "
                                f"(not in the voice set)")
            buffer.append(token)
            continue
        if not _is_clean_boundary("".join(buffer)):
            warnings.append(f"Tag '[{tag}]' sits inside a sentence; the "
                            f"prosody breaks at that boundary")
        flush()
        current_tag = tag
    buffer.append(text[position:])
    flush()

    return Audio8Script(tuple(segments)), warnings
