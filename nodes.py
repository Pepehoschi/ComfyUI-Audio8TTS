from pathlib import Path

import folder_paths

from .audio8_paths import local_model_path
from .audio8_runtime import load_model, encode_reference, generate
from .voice_presets import save_voice, load_voice, voice_signature


class Audio8ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dtype": (["auto", "bfloat16", "float16", "float32", "int8_convrot"], {
                "tooltip": "int8_convrot loads bfloat16 activations and stores the language "
                           "model's nn.Linear weights as int8 with per-group rotation "
                           "(comfy-kitchen TensorWiseINT8Layout, group 64); in_features not "
                           "divisible by 64 stay float. Measured 803 -> 403 MiB of Linear "
                           "weights with audibly equivalent output on the standalone path; "
                           "generation is slower and was only verified without cuda_graphs "
                           "and with comfy_ops off (comfy_ops is ignored with a warning for "
                           "quantised models)."
                           " A swapped checkpoint directory is detected and reloads."}),
        }}

    @classmethod
    def IS_CHANGED(cls, dtype="auto"):
        # The model path is fixed per installation, so a replaced checkpoint would
        # otherwise keep serving from the node cache forever. The directory holds a
        # handful of files; stat-ing them per queue check is cheap.
        files = [p for p in local_model_path().rglob("*") if p.is_file()]
        return (dtype, len(files), sum(p.stat().st_size for p in files),
                max((p.stat().st_mtime for p in files), default=0.0))

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
            "retry_max_new_tokens": ("INT", {
                "default": 2000, "min": 0,
                "tooltip": "Token budget for one retry when the first pass found no EOS. "
                           "Upstream caps every request at max_seq_len (2048) minus the "
                           "prompt length; the effective budget is logged when capped."}),
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
                "tooltip": "With cuda_graphs enabled. \"fast\" replays the fast decoder only. \"frame\" replays the whole frame, slow decode step and codebook chain together, measured about four times faster with bit-identical audio; it needs more graph memory and samples different codes for the same seed. Not combined with an attention_backend that drops the decode mask (flash, comfy_kitchen_int8): the node warns and falls back to \"fast\"."}),
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
