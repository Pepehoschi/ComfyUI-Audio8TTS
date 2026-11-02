"""Persistent Audio8 voice references stored as validated safetensors files."""

import ntpath
import hashlib
import os
import tempfile
from typing import Any

from safetensors import safe_open
from safetensors import SafetensorError
from safetensors.torch import save_file

from .audio8_types import (
    Audio8Reference,
    CODEBOOK_SIZE,
    CODEC_SAMPLE_RATE,
    FORMAT_VERSION,
    MODEL_ID,
    NUM_CODEBOOKS,
    validate_reference,
)


FORMAT_NAME = "comfyui_audio8_voice"
FILE_EXTENSION = ".safetensors"


def _voice_filename(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Voice preset name must be a string")

    name = name.strip()
    if name.lower().endswith(FILE_EXTENSION):
        name = name[: -len(FILE_EXTENSION)]
    if not name or name in {".", ".."}:
        raise ValueError("Voice preset name must not be empty")
    if os.path.isabs(name) or ntpath.isabs(name) or ntpath.splitdrive(name)[0]:
        raise ValueError("Voice preset name must be a relative filename")
    if "/" in name or "\\" in name:
        raise ValueError("Voice preset name must not contain path separators")
    if any(char in '<>:"|?*' or ord(char) < 32 for char in name) or name.endswith((".", " ")):
        raise ValueError("Voice preset name contains invalid Windows filename characters")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if name.split(".")[0].upper() in reserved:
        raise ValueError("Voice preset name is reserved on Windows")
    if any(part in {".", ".."} for part in ntpath.split(name)):
        raise ValueError("Voice preset name must not contain path traversal")
    return name + FILE_EXTENSION


def _voice_path(directory: str, name: str, *, create_directory: bool) -> str:
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("Voice preset directory must be a path")
    base = os.path.realpath(os.fspath(directory))
    if create_directory:
        os.makedirs(base, exist_ok=True)
    filename = _voice_filename(name)
    path = os.path.realpath(os.path.join(base, filename))
    if os.path.commonpath((base, path)) != base:
        raise ValueError("Voice preset path escapes its directory")
    return path


def _metadata(reference: Audio8Reference) -> dict[str, str]:
    return {
        "format": FORMAT_NAME,
        "format_version": str(FORMAT_VERSION),
        "model_id": MODEL_ID,
        "reference_text": reference.reference_text,
        "codec_sample_rate": str(CODEC_SAMPLE_RATE),
        "num_codebooks": str(NUM_CODEBOOKS),
        "codebook_size": str(CODEBOOK_SIZE),
        "frames": str(reference.codes.shape[1]),
    }


def save_voice(reference: Audio8Reference, directory: str, name: str) -> str:
    """Validate and atomically save a voice reference, returning its path."""

    validate_reference(reference)
    path = _voice_path(directory, name, create_directory=True)
    temporary_path: str | None = None
    try:
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=".audio8-voice-",
            suffix=FILE_EXTENSION,
            dir=os.path.dirname(path),
        )
        os.close(file_descriptor)
        save_file(
            {"codes": reference.codes.contiguous()},
            temporary_path,
            metadata=_metadata(reference),
        )
        os.replace(temporary_path, path)
        temporary_path = None
        return path
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _metadata_int(metadata: dict[str, str], key: str) -> int:
    value: Any = metadata.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Voice preset metadata '{key}' is missing or not a string")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Voice preset metadata '{key}' must be an integer") from exc


def load_voice(directory: str, name: str) -> Audio8Reference:
    """Load and strictly validate a voice reference from a safetensors file."""

    path = _voice_path(directory, name, create_directory=False)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio8 voice preset not found: {name}")

    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if metadata is None:
                raise ValueError("Voice preset is missing metadata")
            codes = handle.get_tensor("codes")
    except KeyError as exc:
        raise ValueError("Voice preset is missing the 'codes' tensor") from exc
    except ValueError:
        raise
    except (SafetensorError, OSError) as exc:
        raise ValueError(f"Could not read Audio8 voice preset '{name}'") from exc

    if metadata.get("format") != FORMAT_NAME:
        raise ValueError("Unsupported Audio8 voice preset format")
    if _metadata_int(metadata, "format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported Audio8 voice preset format version")
    if metadata.get("model_id") != MODEL_ID:
        raise ValueError("Voice preset was created for an incompatible Audio8 model")
    reference_text = metadata.get("reference_text")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise ValueError("Voice preset is missing reference_text")

    codec_sample_rate = _metadata_int(metadata, "codec_sample_rate")
    num_codebooks = _metadata_int(metadata, "num_codebooks")
    codebook_size = _metadata_int(metadata, "codebook_size")
    frames = _metadata_int(metadata, "frames")
    if codes.ndim != 2 or codes.shape[1] != frames:
        raise ValueError("Voice preset frames metadata does not match the codes tensor")

    reference = Audio8Reference(
        codes=codes,
        reference_text=reference_text,
        model_id=metadata["model_id"],
        codec_sample_rate=codec_sample_rate,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        format_version=FORMAT_VERSION,
    )
    return validate_reference(reference)


def voice_signature(directory: str, name: str) -> str:
    with open(_voice_path(directory, name, create_directory=False), "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
