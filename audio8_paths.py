"""Path handling: packs from the host's diffusion_models folders.

Single-file packs (*.safetensors) are the only loadable model format. They
are picked from ComfyUI's diffusion_models folders - the same mapping and
picker the native Load Diffusion Model node uses - and carry every weight,
config and tokenizer inside one file (see audio8_pack).

Model directories remain in play only as the source for the Audio8 Pack
Model converter; they are scanned under the registered audio8_tts folders.
"""

from pathlib import Path

import folder_paths


MODEL_DIRECTORY = "Audio8-TTS-Realistic-0.6b"
PACK_SUBFOLDER = "audio8"
PACK_SUFFIX = ".safetensors"


def register_folders():
    # audio8_tts keeps holding converter source directories (and the voice
    # folder holds presets); packs need no registration because
    # diffusion_models is a built-in ComfyUI folder type.
    for name in ("audio8_tts", "audio8_voices"):
        folder_paths.add_model_folder_path(name, str(Path(folder_paths.models_dir) / name))


def _roots():
    return [Path(root) for root in folder_paths.get_folder_paths("audio8_tts")]


def _candidate(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def available_models(max_depth: int = 2) -> list[str]:
    """List Audio8 model directories under every registered audio8_tts root.

    Converter sources only: the loader offers packs, not directories. A
    candidate is a directory holding config.json, searched max_depth levels
    deep so both <root>/Model/ and <root>/Vendor/Model/ are found. Roots
    come from folder_paths, so extra_model_paths.yaml entries participate
    exactly like for ComfyUI's own loaders.
    """

    found = {}
    for root in _roots():
        if not root.is_dir():
            continue
        frontier = [(root, 0)]
        while frontier:
            directory, depth = frontier.pop()
            try:
                children = sorted(p for p in directory.iterdir() if p.is_dir())
            except OSError:
                continue
            for child in children:
                if _candidate(child):
                    found.setdefault(str(child.relative_to(root)).replace("\\", "/"), child)
                elif depth + 1 < max_depth:
                    frontier.append((child, depth + 1))
    return sorted(found)


def _safe_join(root: Path, model_name: str) -> Path:
    relative = Path(model_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"[Audio8] Invalid model name: {model_name!r}")
    return root / relative


def _require_complete(candidate: Path) -> Path:
    if not (candidate / "codec.pth").is_file():
        raise FileNotFoundError(f"[Audio8] Missing codec.pth in {candidate}")
    return candidate


def local_model_dir(model_name: str = "auto", model_path: str = "") -> Path:
    """Resolve a model directory as an Audio8 Pack Model converter source.

    Explicit absolute paths win; otherwise 'auto' prefers the default
    directory name and falls back to the single discovered candidate.
    """

    if model_path and model_path.strip():
        candidate = Path(model_path.strip().strip('"'))
        if not candidate.is_absolute():
            raise FileNotFoundError(
                "[Audio8] model_path must be an absolute path to a model "
                "directory, e.g. "
                "D:\\SharedModels\\Audio8\\Audio8-TTS-Realistic-0.6b (the "
                "folder containing config.json and codec.pth).")
        if not candidate.is_dir():
            raise FileNotFoundError(f"[Audio8] Model directory not found: {candidate}")
        missing = [name for name in ("config.json", "codec.pth")
                   if not (candidate / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"[Audio8] {candidate} is not a complete Audio8 model directory; "
                f"missing: {', '.join(missing)}")
        return candidate

    if model_name and model_name != "auto":
        for root in _roots():
            candidate = _safe_join(root, model_name)
            if _candidate(candidate):
                return _require_complete(candidate)
        raise FileNotFoundError(
            f"[Audio8] Model '{model_name}' not found under any registered "
            "audio8_tts folder (models/audio8_tts or extra_model_paths.yaml).")

    incomplete = []
    for root in _roots():
        candidate = root / MODEL_DIRECTORY
        if (candidate / "config.json").is_file():
            if not (candidate / "codec.pth").is_file():
                incomplete.append(str(candidate))
                continue
            return candidate
    if incomplete:
        raise FileNotFoundError(f"[Audio8] Missing codec.pth in: {', '.join(incomplete)}")

    options = available_models()
    if len(options) == 1:
        for root in _roots():
            option = _safe_join(root, options[0])
            if _candidate(option):
                return _require_complete(option)
    elif options:
        raise FileNotFoundError(
            "Multiple Audio8 model directories available; pick one with the "
            f"model_name input: {', '.join(options)}")
    raise FileNotFoundError(
        f"[Audio8] Place the complete local model in models/audio8_tts/{MODEL_DIRECTORY}/")


def resolve_pack(ckpt_name: str | None) -> Path:
    """Resolve a selected pack via the host's diffusion_models folders.

    Uses folder_paths.get_full_path, the same lookup the native loaders
    apply, so extra_model_paths.yaml roots and subfolder names behave
    identically; traversal attempts are rejected by that lookup.
    """

    if not ckpt_name or not str(ckpt_name).strip():
        raise ValueError(
            "[Audio8] ckpt_name must select an Audio8 pack (*.safetensors) "
            "from the diffusion_models folders; create one from a model "
            "directory with the Audio8 Pack Model node.")
    full_path = folder_paths.get_full_path("diffusion_models", str(ckpt_name).strip())
    if not full_path:
        raise FileNotFoundError(
            f"[Audio8] Pack '{ckpt_name}' not found under the diffusion_models "
            "folders (models/diffusion_models plus extra_model_paths.yaml entries).")
    pack_path = Path(full_path)
    if not pack_path.is_file():
        raise FileNotFoundError(f"[Audio8] Pack file not found: {pack_path}")
    if pack_path.suffix.lower() != PACK_SUFFIX:
        raise ValueError(
            f"[Audio8] {pack_path} is not an Audio8 pack: expected a "
            f"{PACK_SUFFIX} file created by the Audio8 Pack Model node. "
            "Model directory loading was removed; convert once and delete "
            "nothing else.")
    return pack_path


def pack_target_path(pack_name: str) -> Path:
    """Target for new packs: audio8/ under the diffusion_models root that
    already carries an audio8/ folder; falls back to the first root.

    get_folder_paths lists built-in defaults (models/unet aliases
    diffusion_models) before extra_model_paths.yaml roots, so picking
    roots[0] would split old and new packs across two trees. Writing next
    to the existing packs keeps one audio8/ folder per install.
    """

    roots = folder_paths.get_folder_paths("diffusion_models")
    if not roots:
        raise FileNotFoundError(
            "[Audio8] No diffusion_models folder is registered; ComfyUI "
            "normally provides models/diffusion_models.")
    name = str(pack_name).strip().strip('"')
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"[Audio8] Invalid pack_name: {pack_name!r}")
    if name.lower().endswith(PACK_SUFFIX):
        name = name[: -len(PACK_SUFFIX)]
    target_root = next((root for root in roots
                        if (Path(root) / PACK_SUBFOLDER).is_dir()), roots[0])
    return Path(target_root) / PACK_SUBFOLDER / (name + PACK_SUFFIX)
