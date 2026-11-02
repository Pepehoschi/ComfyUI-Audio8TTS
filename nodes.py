from pathlib import Path

import folder_paths

from .audio8_runtime import load_model, encode_reference, generate
from .voice_presets import save_voice, load_voice, voice_signature


class Audio8ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"dtype": (["auto", "bfloat16", "float16", "float32"],)}}

    RETURN_TYPES = ("AUDIO8_MODEL",)
    FUNCTION = "load"
    CATEGORY = "audio/audio8"

    def load(self, dtype="auto"):
        return (load_model(dtype),)


class Audio8EncodeReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("AUDIO8_MODEL",), "audio": ("AUDIO",),
                             "reference_text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("AUDIO8_REFERENCE",)
    FUNCTION = "encode"
    CATEGORY = "audio/audio8"

    def encode(self, model, audio, reference_text):
        return (encode_reference(model, audio, reference_text),)


class Audio8SaveVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reference": ("AUDIO8_REFERENCE",),
                             "name": ("STRING", {"default": "voice"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "audio/audio8"
    OUTPUT_NODE = True

    def save(self, reference, name):
        directory = folder_paths.get_folder_paths("audio8_voices")[0]
        return (save_voice(reference, directory, name),)


class Audio8LoadVoice:
    @classmethod
    def IS_CHANGED(cls, voice):
        for directory in folder_paths.get_folder_paths("audio8_voices"):
            try:
                return voice_signature(directory, voice)
            except FileNotFoundError:
                continue
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        names = set()
        for directory in folder_paths.get_folder_paths("audio8_voices"):
            if Path(directory).is_dir():
                names.update(path.name for path in Path(directory).glob("*.safetensors"))
        return {"required": {"voice": (sorted(names),)}}

    RETURN_TYPES = ("AUDIO8_REFERENCE",)
    FUNCTION = "load"
    CATEGORY = "audio/audio8"

    def load(self, voice):
        for directory in folder_paths.get_folder_paths("audio8_voices"):
            try:
                return (load_voice(directory, voice),)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"[Audio8] Voice preset not found: {voice}")


class Audio8TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("AUDIO8_MODEL",), "text": ("STRING", {"multiline": True}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "max_new_tokens": ("INT", {"default": 1024, "min": 1}),
            "temperature": ("FLOAT", {"default": 0.8, "min": 0.0}),
            "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
            "top_k": ("INT", {"default": 50, "min": 0}),
            "greedy": ("BOOLEAN", {"default": False}),
            "retry_max_new_tokens": ("INT", {"default": 2000, "min": 0}),
            "chunking": (["auto", "disabled"],),
            "target_chunk_chars": ("INT", {"default": 130, "min": 1}),
            "max_chunk_chars": ("INT", {"default": 150, "min": 1}),
            "sentence_pause_ms": ("INT", {"default": 180, "min": 0}),
            "paragraph_pause_ms": ("INT", {"default": 350, "min": 0}),
        }, "optional": {
            "reference": ("AUDIO8_REFERENCE",),
            "fade_out_ms": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 1.0,
                                      "tooltip": "Fade out the end of each chunk before its pause. Set 0 to disable."}),
            "cuda_graphs": ("BOOLEAN", {"default": False,
                                        "tooltip": "Replay the fast decoder with CUDA Graphs. CUDA only; graph setup may outweigh savings on very short text."}),
            "cuda_graph_scope": (["fast", "frame"], {
                "default": "fast",
                "tooltip": "With cuda_graphs enabled. \"fast\" replays the fast decoder only. \"frame\" replays the whole frame, slow decode step and codebook chain together, measured about three times faster; it needs more graph memory and samples different codes for the same seed."}),
            "comfy_ops": ("BOOLEAN", {"default": False,
                                      "tooltip": "Use shared ComfyUI layers, rotary encoding and selected attention backend. Floating-point results can differ from the original model."}),
            "attention_backend": (["auto", "flash", "comfy_kitchen_int8"], {
                "default": "auto",
                "tooltip": "With comfy_ops enabled, choose optimized slow-decoder attention. Requires the selected backend to be installed; auto retains ComfyUI's default."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "synthesize"
    CATEGORY = "audio/audio8"

    def synthesize(self, model, text, **settings):
        return (generate(model, text, **settings),)


NODE_CLASS_MAPPINGS = {
    "Audio8ModelLoader": Audio8ModelLoader,
    "Audio8EncodeReference": Audio8EncodeReference,
    "Audio8SaveVoice": Audio8SaveVoice,
    "Audio8LoadVoice": Audio8LoadVoice,
    "Audio8TTS": Audio8TTS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio8ModelLoader": "Audio8 Model Loader",
    "Audio8EncodeReference": "Audio8 Encode Reference",
    "Audio8SaveVoice": "Audio8 Save Voice",
    "Audio8LoadVoice": "Audio8 Load Voice",
    "Audio8TTS": "Audio8 TTS",
}
