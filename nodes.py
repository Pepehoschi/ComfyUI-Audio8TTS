from pathlib import Path
import logging

import folder_paths

from .audio8_paths import (PACK_SUBFOLDER, available_models, local_model_dir,
                           pack_target_path, resolve_pack)
from .audio8_pack import pack_model_directory
from .audio8_runtime import load_model, encode_reference, generate
from .audio8_script import parse_script
from .audio8_types import VoiceSet
from .voice_presets import save_voice, load_voice, voice_signature


logger = logging.getLogger("audio8")


class Audio8ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            packs = folder_paths.get_filename_list("diffusion_models")
        except Exception:
            packs = []
        return {"required": {
            "ckpt_name": (packs, {
                "tooltip": "Audio8 packs (*.safetensors) from the diffusion_models "
                           "folders - models/diffusion_models plus every "
                           "extra_model_paths.yaml entry, the same folders the "
                           "native Load Diffusion Model node lists. Create a pack "
                           "from an Audio8 model directory once with the "
                           "Audio8 Pack Model node."}),
            "dtype": ([ "auto", "bfloat16", "float16", "float32", "int8_convrot"], {
                "tooltip": "int8_convrot loads bfloat16 activations and stores the language "
                           "model's nn.Linear weights as int8 with per-group rotation "
                           "(comfy-kitchen TensorWiseINT8Layout, group 64); in_features not "
                           "divisible by 64 stay float. Measured 803 -> 403 MiB of Linear "
                           "weights with audibly equivalent output on the standalone path; "
                           "generation is slower and was only verified without cuda_graphs "
                           "and with comfy_ops off (comfy_ops is ignored with a warning for "
                           "quantised models)."
                           " A swapped pack file is detected and reloads."}),
        }}

    @classmethod
    def IS_CHANGED(cls, ckpt_name="", dtype="auto"):
        # A replaced pack file would otherwise keep serving from the node
        # cache forever; stat-ing the single file per queue check is cheap.
        try:
            target = resolve_pack(ckpt_name)
            stat = target.stat()
            return (dtype, ckpt_name, stat.st_size, stat.st_mtime)
        except Exception:
            # Nothing resolvable (removed or not yet packed): the name alone
            # still invalidates the cache when it changes.
            return (dtype, ckpt_name, "unresolved")

    RETURN_TYPES = ("AUDIO8_MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("Loaded Audio8 model, ready for Audio8 TTS and "
                       "Audio8 Encode Reference.",)
    FUNCTION = "load"
    CATEGORY = "audio/audio8"

    def load(self, ckpt_name, dtype="auto"):
        return (load_model(dtype, ckpt_name),)


class Audio8PackModel:
    """Fold an Audio8 model directory into one .safetensors pack file."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            models = available_models()
        except Exception:
            models = []
        return {"optional": {
            "model_name": (["auto"] + [m for m in models if m != "auto"], {
                "default": "auto",
                "tooltip": "Source model directory (must contain config.json, "
                           "codec.pth and model.safetensors). Packs are not a "
                           "valid source; they are the output of this node."}),
            "model_path": ("STRING", {
                "default": "",
                "tooltip": "Absolute path to the source model directory. Takes "
                           "precedence over model_name."}),
            "pack_name": ("STRING", {
                "default": "",
                "tooltip": "File name for the pack inside the audio8/ subfolder "
                           "of the first diffusion_models folder (without "
                           ".safetensors). Empty uses the source directory "
                           "name."}),
            "overwrite": ("BOOLEAN", {
                "default": False,
                "tooltip": "Replace an existing pack file of the same name."}),
            "quantize": (["none", "int8_convrot"], {
                "default": "none",
                "tooltip": "int8_convrot writes a separate pack with int8 "
                           "ConvRot weights (audio8_quant=int8_convrot "
                           "metadata); the file name then gets an "
                           "-int8convrot suffix unless pack_name overrides it."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pack_name",)
    OUTPUT_TOOLTIPS = ("Pack name relative to the diffusion_models folders, "
                       "directly selectable in Audio8 Model Loader.",)
    FUNCTION = "pack"
    CATEGORY = "audio/audio8"
    OUTPUT_NODE = True

    def pack(self, model_name="auto", model_path="", pack_name="", overwrite=False,
             quantize="none"):
        source = local_model_dir(model_name, model_path)
        name = pack_name or source.name
        if quantize == "int8_convrot" and not pack_name:
            name = f"{name}-int8convrot"
        target = pack_target_path(name)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"[Audio8] Pack {target} already exists; set overwrite to replace it.")
        target.parent.mkdir(parents=True, exist_ok=True)
        pack_model_directory(source, target, overwrite=bool(overwrite), quantize=quantize)
        return (f"{PACK_SUBFOLDER}/{target.name}",)


class Audio8EncodeReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("AUDIO8_MODEL", {"tooltip":
                "Loaded Audio8 model; its codec turns the reference audio "
                "into codes."}),
            "audio": ("AUDIO", {"tooltip":
                "Reference recording whose voice character is cloned. Short "
                "and clean works best; any sample rate, it is resampled to "
                "the codec rate."}),
            "reference_text": ("STRING", {"multiline": True, "tooltip":
                "Exact transcript of what is spoken in the reference audio. "
                "It is stored with the codes and acts as the speaking "
                "prompt."}),
        }}

    RETURN_TYPES = ("AUDIO8_REFERENCE",)
    RETURN_NAMES = ("reference",)
    OUTPUT_TOOLTIPS = ("Reference preset (codec codes plus transcript); save "
                       "it with Audio8 Save Voice or feed it straight into "
                       "Audio8 TTS.",)
    FUNCTION = "encode"
    CATEGORY = "audio/audio8"

    def encode(self, model, audio, reference_text):
        return (encode_reference(model, audio, reference_text),)


class Audio8SaveVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference": ("AUDIO8_REFERENCE", {"tooltip":
                "Reference preset from Audio8 Encode Reference."}),
            "name": ("STRING", {"default": "voice", "tooltip":
                "File name inside the audio8_voices folders (without "
                ".safetensors); Audio8 Load Voice lists it."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    OUTPUT_TOOLTIPS = ("Full path of the written preset file.",)
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
        return {"required": {"voice": (sorted(names), {"tooltip":
            "Presets (*.safetensors) from the audio8_voices folders; a "
            "replaced file is detected and reloads."})}}

    RETURN_TYPES = ("AUDIO8_REFERENCE",)
    RETURN_NAMES = ("reference",)
    OUTPUT_TOOLTIPS = ("Reference preset for Audio8 TTS or for a tag entry "
                       "in a voice-set chain.",)
    FUNCTION = "load"
    CATEGORY = "audio/audio8"

    def load(self, voice):
        for directory in folder_paths.get_folder_paths("audio8_voices"):
            try:
                return (load_voice(directory, voice),)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"[Audio8] Voice preset not found: {voice}")


class Audio8VoiceSet:
    """Empty root of a voice-set chain: free {tag -> reference} dictionary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("AUDIO8_VOICE_SET",)
    RETURN_NAMES = ("voice_set",)
    OUTPUT_TOOLTIPS = ("Empty voice set; register tags with Audio8 Voice "
                       "Set Add.",)
    FUNCTION = "create"
    CATEGORY = "audio/audio8"

    def create(self):
        return (VoiceSet(),)


class Audio8VoiceSetAdd:
    """Append one tag -> voice entry to a voice set.

    Tags are free strings with no relationship to the voice on purpose; the
    only rule is uniqueness: a duplicate gets _1, _2, ... in chain order.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "voice_set": ("AUDIO8_VOICE_SET", {"tooltip":
                "Chain start (Audio8 Voice Set) or the output of the "
                "previous Audio8 Voice Set Add."}),
            "voice": ("AUDIO8_REFERENCE", {"tooltip":
                "Reference preset registered under the tag."}),
            "tag": ("STRING", {"default": "", "tooltip":
                "Tag string used in scripts as [tag]. Any text without square "
                "brackets; unrelated to the voice file name on purpose. A "
                "duplicate tag is registered with a _1, _2 ... suffix "
                "(reported in the log output)."}),
        }}

    RETURN_TYPES = ("AUDIO8_VOICE_SET", "STRING")
    RETURN_NAMES = ("voice_set", "log")
    OUTPUT_TOOLTIPS = ("Updated chain including the new entry; feed into the "
                       "next Add or into Audio8 Script.",
                       "Registration note, e.g. a duplicate tag moved to "
                       "tag_1.")
    FUNCTION = "add"
    CATEGORY = "audio/audio8"

    def add(self, voice_set, voice, tag):
        updated, note = voice_set.with_entry(tag, voice)
        log = note or f"Tag '{tag.strip()}' added to the voice set."
        if note:
            logger.warning("[Audio8] %s", note)
        return (updated, log)


class Audio8Script:
    """Parse tagged text once against a voice set into renderable segments.

    Grammar: [tag] switches voice until the next tag, [[ and ]] escape
    literal brackets, unknown brackets stay in the text and are reported.
    The log output lists every warning; the parser never raises for tags.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "",
                                "tooltip": "Script text. [tag] switches the "
                                           "voice reference from the wired "
                                           "voice set; text before the first "
                                           "tag uses the TTS default "
                                           "reference. [[ and ]] write "
                                           "literal brackets."}),
            "voice_set": ("AUDIO8_VOICE_SET", {"tooltip":
                "Voice-set chain that decides which [tags] are known. "
                "Unknown brackets stay in the text and are reported."}),
        }}

    RETURN_TYPES = ("AUDIO8_SCRIPT", "STRING")
    RETURN_NAMES = ("script", "log")
    OUTPUT_TOOLTIPS = ("Parsed segments for the script input of Audio8 TTS.",
                       "Segment and tag summary plus every parse warning "
                       "(unknown brackets, tags inside a sentence).")
    FUNCTION = "parse"
    CATEGORY = "audio/audio8"

    def parse(self, text, voice_set):
        script, warnings = parse_script(text, voice_set)
        for warning in warnings:
            logger.warning("[Audio8] Script: %s", warning)
        summary = (f"segments: {len(script.segments)}; tags: "
                   f"{', '.join(voice_set.tags()) or '(empty set)'}")
        log = "\n".join([summary] + warnings)
        return (script, log)


class Audio8TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("AUDIO8_MODEL", {"tooltip":
                "Loaded Audio8 model from Audio8 Model Loader."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff,
                             "tooltip": "One generator advances through the "
                                        "whole run; a scripted render keeps "
                                        "one seed across all segments."}),
            "max_new_tokens": ("INT", {"default": 1024, "min": 1,
                                       "tooltip": "Token budget for the "
                                                  "first pass of each chunk."}),
            "temperature": ("FLOAT", {"default": 0.8, "min": 0.0,
                                      "tooltip": "Sampling temperature; "
                                                 "ignored with greedy."}),
            "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0,
                                "tooltip": "Nucleus cutoff; ignored with "
                                           "greedy."}),
            "top_k": ("INT", {"default": 50, "min": 0,
                              "tooltip": "Candidate codes per step; 0 "
                                         "disables. Ignored with greedy."}),
            "greedy": ("BOOLEAN", {"default": False,
                                   "tooltip": "Always pick the most likely "
                                              "code; overrides temperature, "
                                              "top_p and top_k."}),
            "retry_max_new_tokens": ("INT", {
                "default": 2000, "min": 0,
                "tooltip": "Token budget for one retry when the first pass found no EOS. "
                           "Upstream caps every request at max_seq_len (2048) minus the "
                           "prompt length; the effective budget is logged when capped."}),
            "chunking": (["auto", "disabled"], {"tooltip":
                "auto splits long text into sentence-shaped chunks; "
                "disabled sends the whole text in one request."}),
            "target_chunk_chars": ("INT", {"default": 130, "min": 1,
                                           "tooltip": "Split points prefer "
                                                      "sentence ends up to "
                                                      "this length."}),
            "max_chunk_chars": ("INT", {"default": 150, "min": 1,
                                        "tooltip": "Hard cap per chunk; "
                                                   "longer text is split "
                                                   "regardless of sentence "
                                                   "ends."}),
            "sentence_pause_ms": ("INT", {"default": 180, "min": 0,
                                          "tooltip": "Silence after a chunk "
                                                     "ending on a sentence."}),
            "paragraph_pause_ms": ("INT", {"default": 350, "min": 0,
                                           "tooltip": "Longer silence at "
                                                      "paragraph boundaries; "
                                                      "scripted tag switches "
                                                      "use it too."}),
        }, "optional": {
            "text": ("STRING", {"multiline": True, "tooltip":
                "Plain text to speak. Ignored with a warning when the "
                "script input is connected."}),
            "reference": ("AUDIO8_REFERENCE", {"tooltip":
                "Default voice reference (Audio8 Load Voice): renders text "
                "before the first tag and every untagged segment."}),
            "fade_out_ms": ("FLOAT", {"default": 110.0, "min": 0.0, "step": 1.0,
                                      "tooltip": "Fade out the end of each chunk before its pause. Set 0 to disable."}),
            "cuda_graphs": ("BOOLEAN", {"default": True,
                                        "tooltip": "Replay the decoder with CUDA Graphs. CUDA only, ignored on CPU; measured about four times faster with bit-identical audio in the default scope."}),
            "cuda_graph_scope": (["fast", "frame"], {
                "default": "frame",
                "tooltip": "With cuda_graphs enabled. \"fast\" replays the fast decoder only. \"frame\" replays the whole frame, slow decode step and codebook chain together, measured about four times faster with bit-identical audio; it needs more graph memory and samples different codes for the same seed. Not combined with an attention_backend that drops the decode mask (flash, comfy_kitchen_int8): the node warns and falls back to \"fast\"."}),
            "comfy_ops": ("BOOLEAN", {"default": False,
                                      "tooltip": "Use shared ComfyUI layers, rotary encoding and selected attention backend. Floating-point results can differ from the original model."}),
            "attention_backend": (["auto", "flash", "comfy_kitchen_int8"], {
                "default": "auto",
                "tooltip": "With comfy_ops enabled, choose optimized slow-decoder attention. Requires the selected backend to be installed; auto retains ComfyUI's default."}),
            "script": ("AUDIO8_SCRIPT", {
                "tooltip": "Parsed tagged script (Audio8 Script node). When "
                           "connected, text is ignored with a warning and "
                           "each segment renders with its own voice "
                           "reference; segments without a tag fall back to "
                           "the reference input. Listed last so saved "
                           "workflows keep their input slot numbers."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_TOOLTIPS = ("Synthesized waveform at 44100 Hz; connect to Save "
                       "Audio or another AUDIO consumer.",)
    FUNCTION = "synthesize"
    CATEGORY = "audio/audio8"

    def synthesize(self, model, text="", script=None, **settings):
        if script is not None and text and text.strip():
            logger.warning("[Audio8] TTS ignores the text input because a "
                           "script is connected.")
        return (generate(model, text, script=script, **settings),)


NODE_CLASS_MAPPINGS = {
    "Audio8ModelLoader": Audio8ModelLoader,
    "Audio8PackModel": Audio8PackModel,
    "Audio8EncodeReference": Audio8EncodeReference,
    "Audio8SaveVoice": Audio8SaveVoice,
    "Audio8LoadVoice": Audio8LoadVoice,
    "Audio8VoiceSet": Audio8VoiceSet,
    "Audio8VoiceSetAdd": Audio8VoiceSetAdd,
    "Audio8Script": Audio8Script,
    "Audio8TTS": Audio8TTS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio8ModelLoader": "Audio8 Model Loader",
    "Audio8PackModel": "Audio8 Pack Model",
    "Audio8EncodeReference": "Audio8 Encode Reference",
    "Audio8SaveVoice": "Audio8 Save Voice",
    "Audio8LoadVoice": "Audio8 Load Voice",
    "Audio8VoiceSet": "Audio8 Voice Set",
    "Audio8VoiceSetAdd": "Audio8 Voice Set Add",
    "Audio8Script": "Audio8 Script",
    "Audio8TTS": "Audio8 TTS",
}
