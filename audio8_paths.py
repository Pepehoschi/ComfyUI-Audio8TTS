from pathlib import Path

import folder_paths


MODEL_DIRECTORY = "Audio8-TTS-Realistic-0.6b"


def register_folders():
    for name in ("audio8_tts", "audio8_voices"):
        folder_paths.add_model_folder_path(name, str(Path(folder_paths.models_dir) / name))


def local_model_path():
    incomplete = []
    for directory in folder_paths.get_folder_paths("audio8_tts"):
        candidate = Path(directory) / MODEL_DIRECTORY
        if (candidate / "config.json").is_file():
            if not (candidate / "codec.pth").is_file():
                incomplete.append(str(candidate))
                continue
            return candidate
    if incomplete:
        raise FileNotFoundError(f"[Audio8] Missing codec.pth in: {', '.join(incomplete)}")
    raise FileNotFoundError(
        f"[Audio8] Place the complete local model in models/audio8_tts/{MODEL_DIRECTORY}/"
    )
