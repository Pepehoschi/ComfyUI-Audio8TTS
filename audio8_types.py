"""Shared data types and validation for the Audio8 ComfyUI nodes."""

from dataclasses import dataclass
from typing import Any

import torch


MODEL_ID = "Edge0/Audio8-TTS-Realistic-0.6b"
FORMAT_VERSION = 1
CODEC_SAMPLE_RATE = 44100
NUM_CODEBOOKS = 10
CODEBOOK_SIZE = 4096


@dataclass
class Audio8ModelBundle:
    """Loaded Audio8 model and processor reused by downstream nodes."""

    model: Any
    processor: Any
    model_id: str
    device: torch.device
    dtype: torch.dtype
    sample_rate: int
    quantization: str = "none"


@dataclass
class Audio8Reference:
    """Codec codes and transcript required to condition Audio8 generation."""

    codes: torch.Tensor
    reference_text: str
    model_id: str = MODEL_ID
    codec_sample_rate: int = CODEC_SAMPLE_RATE
    num_codebooks: int = NUM_CODEBOOKS
    codebook_size: int = CODEBOOK_SIZE
    format_version: int = FORMAT_VERSION


@dataclass(frozen=True)
class VoiceSet:
    """Ordered, free-form {tag -> reference} dictionary built by node chains.

    Tags are arbitrary user strings; the only invariant is uniqueness, which
    with_entry enforces by suffixing duplicates with _1, _2, ... in chain
    order. There is deliberately no relationship checked between a tag and
    its reference: any voice may carry any tag.
    """

    entries: tuple[tuple[str, Audio8Reference], ...] = ()

    def tags(self) -> tuple[str, ...]:
        return tuple(tag for tag, _ in self.entries)

    def get(self, tag: str) -> Audio8Reference | None:
        for entry_tag, reference in self.entries:
            if entry_tag == tag:
                return reference
        return None

    def with_entry(self, tag: str, reference: Audio8Reference) -> tuple["VoiceSet", str | None]:
        if not isinstance(tag, str):
            raise TypeError("Voice set tag must be a string")
        tag = tag.strip()
        if not tag:
            raise ValueError("Voice set tag must not be empty")
        if "[" in tag or "]" in tag:
            raise ValueError("Voice set tag must not contain square brackets")
        validate_reference(reference)
        used = set(self.tags())
        final_tag = tag
        rename_note = None
        if tag in used:
            index = 1
            while f"{tag}_{index}" in used:
                index += 1
            final_tag = f"{tag}_{index}"
            rename_note = (f"Tag '{tag}' already exists in the voice set; "
                           f"this entry was registered as '{final_tag}'")
        return VoiceSet(self.entries + ((final_tag, reference),)), rename_note


@dataclass(frozen=True)
class ScriptSegment:
    """One renderable span of script text with its resolved voice reference.

    A reference of None means "use the TTS node's default reference input".
    """

    tag: str | None
    reference: Audio8Reference | None
    text: str


@dataclass(frozen=True)
class Audio8Script:
    """Parsed script: ordered segments produced by one parse of the raw text."""

    segments: tuple[ScriptSegment, ...]


def validate_reference(reference: Audio8Reference) -> Audio8Reference:
    """Validate an Audio8 reference and return it for convenient chaining."""

    if not isinstance(reference, Audio8Reference):
        raise TypeError("Audio8 reference must be an Audio8Reference instance")

    codes = reference.codes
    if not isinstance(codes, torch.Tensor):
        raise TypeError("Audio8 reference codes must be a torch.Tensor")
    if codes.device.type != "cpu":
        raise ValueError("Audio8 reference codes must be stored on the CPU")
    if codes.dtype != torch.long:
        raise ValueError("Audio8 reference codes must use torch.long")
    if codes.ndim != 2 or codes.shape[0] != NUM_CODEBOOKS or codes.shape[1] == 0:
        raise ValueError(f"Audio8 reference codes must have shape [{NUM_CODEBOOKS}, T] with T > 0")
    if int(codes.min()) < 0 or int(codes.max()) >= CODEBOOK_SIZE:
        raise ValueError(f"Audio8 reference codes must be in the range 0..{CODEBOOK_SIZE - 1}")

    if not isinstance(reference.reference_text, str) or not reference.reference_text.strip():
        raise ValueError("Audio8 reference_text must be a non-empty string")
    if reference.model_id != MODEL_ID:
        raise ValueError(f"Unsupported Audio8 model_id: {reference.model_id!r}")
    if reference.codec_sample_rate != CODEC_SAMPLE_RATE:
        raise ValueError(f"Audio8 codec_sample_rate must be {CODEC_SAMPLE_RATE}")
    if reference.num_codebooks != NUM_CODEBOOKS:
        raise ValueError(f"Audio8 num_codebooks must be {NUM_CODEBOOKS}")
    if reference.codebook_size != CODEBOOK_SIZE:
        raise ValueError(f"Audio8 codebook_size must be {CODEBOOK_SIZE}")
    if reference.format_version != FORMAT_VERSION:
        raise ValueError(f"Unsupported Audio8 reference format_version: {reference.format_version!r}")

    return reference
