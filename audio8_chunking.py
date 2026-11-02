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
            # Cut-point order: a strong boundary (sentence or subsentence) at or
            # beyond the target, then a strong one that is at least half the
            # target, then any boundary at or beyond the target. Only when the
            # window offers none of those fall back to the strongest boundary
            # anywhere; a strong boundary far before the target must not steal
            # the cut from a boundary near the target ("v1. " + 400 words).
            strong = [(offset, priority) for offset, priority in available if priority >= 2]
            far_strong = [(offset, priority) for offset, priority in strong if offset >= target]
            decent = [offset for offset, _ in strong if offset * 2 >= target]
            far_any = [offset for offset, _ in available if offset >= target]
            if far_strong:
                best_priority = max(priority for _, priority in far_strong)
                cut = min(offset for offset, priority in far_strong if priority == best_priority)
            elif decent:
                cut = max(decent)
            elif far_any:
                cut = min(far_any)
            else:
                highest_priority = max(priority for _, priority in available)
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
