"""Semantic text chunking for Audio8 generation."""

from dataclasses import dataclass
import re


SENTENCE_ENDINGS = frozenset(".!?…。！？؟")
SUBSENTENCE_BREAKS = frozenset(";,：:，、—–")


@dataclass(frozen=True)
class TextChunk:
    text: str
    pause_after_ms: int


def _normalise_paragraph(text: str) -> str:
    return " ".join(text.split())


def _boundaries(text: str) -> list[tuple[int, int]]:
    """Return end offsets and priorities, where larger priority is preferred."""

    result = []
    for index, character in enumerate(text):
        end = index + 1
        if character in SENTENCE_ENDINGS:
            result.append((end, 3))
        elif character in SUBSENTENCE_BREAKS:
            result.append((end, 2))
        elif character.isspace():
            result.append((end, 1))
    return result


def _split_paragraph(text: str, target: int, maximum: int) -> list[str]:
    chunks = []
    remaining = text
    while len(remaining) > maximum:
        available = _boundaries(remaining[:maximum])
        if available:
            highest_priority = max(priority for _, priority in available)
            preferred = [
                offset for offset, priority in available
                if priority == highest_priority and offset >= target
            ]
            if preferred:
                cut = min(preferred)
            else:
                cut = max(offset for offset, priority in available if priority == highest_priority)
        else:
            cut = maximum

        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].lstrip()

    if remaining:
        chunks.append(remaining.strip())
    return chunks


def split_text(
    text: str,
    target_chunk_chars: int = 130,
    max_chunk_chars: int = 150,
    sentence_pause_ms: int = 180,
    paragraph_pause_ms: int = 350,
) -> list[TextChunk]:
    """Split text at natural boundaries while preserving all non-whitespace text."""

    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    if target_chunk_chars <= 0 or max_chunk_chars <= 0:
        raise ValueError("Chunk character limits must be positive")
    maximum = max_chunk_chars
    target = min(target_chunk_chars, maximum)

    paragraphs = [_normalise_paragraph(part) for part in re.split(r"\n\s*\n", text)]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if not paragraphs:
        return []

    result = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        pieces = _split_paragraph(paragraph, target, maximum)
        for piece_index, piece in enumerate(pieces):
            is_last_piece = piece_index == len(pieces) - 1
            is_last_paragraph = paragraph_index == len(paragraphs) - 1
            if is_last_piece and not is_last_paragraph:
                pause = paragraph_pause_ms
            elif is_last_piece:
                pause = 0
            else:
                pause = sentence_pause_ms
            result.append(TextChunk(piece, pause))
    return result
