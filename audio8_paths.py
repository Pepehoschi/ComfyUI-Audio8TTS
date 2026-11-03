from pathlib import Path

import folder_paths


MODEL_DIRECTORY = "Audio8-TTS-Realistic-0.6b"
ARCH_MIRROR_SUFFIX = "__with_arch"


def register_folders():
    for name in ("audio8_tts", "audio8_voices"):
        folder_paths.add_model_folder_path(name, str(Path(folder_paths.models_dir) / name))


def _roots():
    return [Path(root) for root in folder_paths.get_folder_paths("audio8_tts")]


def _candidate(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def available_models(max_depth: int = 2) -> list[str]:
    """List Audio8 model directories under every registered audio8_tts root.

    A candidate is a directory holding config.json, searched max_depth levels
    deep so both <root>/Model/ and <root>/Vendor/Model/ are found. Roots come
    from folder_paths, so extra_model_paths.yaml entries participate exactly
    like for ComfyUI's own loaders.
    """

    found = {}
    for root in _roots():
        if not root.is_dir():
            continue
        frontier = [(root, 0)]
        while frontier:
            directory, depth = frontier.pop()
            if directory.name.endswith(ARCH_MIRROR_SUFFIX):
                continue
            try:
                children = sorted(p for p in directory.iterdir() if p.is_dir())
            except OSError:
                continue
            for child in children:
                if child.name.endswith(ARCH_MIRROR_SUFFIX):
                    continue
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


def available_packs(max_depth: int = 2) -> list[str]:
    """List single-file Audio8 packs (*.safetensors) under registered roots.

    Mirrors available_models() for the one-file pack format created by the
    Audio8 Pack Model node, so packs appear in the same selection dropdown
    as model directories.
    """

    found = {}
    for root in _roots():
        if not root.is_dir():
            continue
        frontier = [(root, 0)]
        while frontier:
            directory, depth = frontier.pop()
            if directory.name.endswith(ARCH_MIRROR_SUFFIX):
                continue
            try:
                children = sorted(directory.iterdir())
            except OSError:
                continue
            for child in children:
                if child.name.endswith(ARCH_MIRROR_SUFFIX):
                    continue
                if child.is_file() and child.suffix.lower() == ".safetensors":
                    found.setdefault(str(child.relative_to(root)).replace("\\", "/"), child)
                elif child.is_dir() and depth + 1 < max_depth and not _candidate(child):
                    # never descend into a model directory: its inner
                    # model.safetensors belongs to the directory entry
                    frontier.append((child, depth + 1))
    return sorted(found)


def _resolve(root: Path, relative: str) -> Path:
    candidate = _safe_join(root, relative)
    if not (candidate / "codec.pth").is_file():
        raise FileNotFoundError(f"[Audio8] Missing codec.pth in {candidate}")
    return candidate


def local_model_path(model_name: str = "auto", model_path: str = "") -> Path:
    """Resolve a model directory or pack file; explicit paths win."""

    if model_path and model_path.strip():
        candidate = Path(model_path.strip().strip('"'))
        if not candidate.is_absolute():
            raise FileNotFoundError(
                "[Audio8] model_path must be an absolute path to a model "
                "directory or an Audio8 .safetensors pack, e.g. "
                "D:\\SharedModels\\Audio8\\Audio8-TTS-Realistic-0.6b (the folder "
                "containing config.json and codec.pth). For registered audio8_tts "
                "folders use model_name instead.")
        if candidate.suffix.lower() == ".safetensors":
            if not candidate.is_file():
                raise FileNotFoundError(f"[Audio8] Pack file not found: {candidate}")
            return candidate
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
        if model_name.lower().endswith(".safetensors"):
            for root in _roots():
                candidate = _safe_join(root, model_name)
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(
                f"[Audio8] Pack '{model_name}' not found under any registered "
                "audio8_tts folder (models/audio8_tts or extra_model_paths.yaml).")
        for root in _roots():
            candidate = _safe_join(root, model_name)
            if _candidate(candidate):
                return _resolve(root, model_name)
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

    options = available_models() + available_packs()
    if len(options) == 1:
        for root in _roots():
            option = _safe_join(root, options[0])
            if options[0].lower().endswith(".safetensors"):
                if option.is_file():
                    return option
            elif _candidate(option):
                return _resolve(root, options[0])
    elif options:
        raise FileNotFoundError(
            "[Audio8] Multiple Audio8 models available; pick one with the "
            f"model_name input: {', '.join(options)}")
    raise FileNotFoundError(
        f"[Audio8] Place the complete local model in models/audio8_tts/{MODEL_DIRECTORY}/"
    )
