"""Single-file Audio8 pack format (.safetensors).

A pack folds one Audio8 model directory into one file, the way ComfyUI's
own checkpoint loaders work: language-model tensors under "model.", codec
tensors under "codec.", and the config/tokenizer JSONs as safetensors
header metadata. Loading instantiates the node's bundled Apache-2.0
architecture (audio8_arch) directly and fills it via load_state_dict, so
no remote code, no trust_remote_code and no Auto* registration are used
(transformers 5.x silently skips materializing checkpoint weights for
registered custom classes - verified on 5.9.0).

Format v1 metadata keys:
    audio8_format            -> "1"
    audio8_quant             -> "none" or "int8_convrot"
    audio8_int8              -> json map module path -> int8 Params fields,
                                present only when audio8_quant=int8_convrot;
                                the quantised weights keep the plain
                                "<path>.weight" key as int8 tensors and gain
                                a "<path>.weight_scale" float32 companion
    audio8_model_id          -> upstream model id
    audio8_created           -> ISO-8601 UTC creation stamp
    audio8_config            -> config.json text
    audio8_generation_config -> generation_config.json text (informational)
    audio8_processor_config  -> processor_config.json text
    audio8_tokenizer_config  -> tokenizer_config.json text
    audio8_special_tokens_map-> special_tokens_map.json text
    audio8_tokenizer_json    -> tokenizer.json text (includes added tokens)
    audio8_source_arch_sha   -> json map of the source directory's architecture
                                file sha256 values, when the directory shipped
                                them (audit trail against the bundled copy)
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from .audio8_quantization import INT8_CONVROT, INT8_CONVROT_GROUP_SIZE
from .audio8_types import MODEL_ID


logger = logging.getLogger(__name__)

PACK_SUFFIX = ".safetensors"
PACK_FORMAT = "1"
PACK_EXTENSION = "none"
PACK_QUANTIZATIONS = (PACK_EXTENSION, INT8_CONVROT)
MODEL_PREFIX = "model."
CODEC_PREFIX = "codec."
INT8_SCALE_SUFFIX = ".weight_scale"

# metadata key -> file name inside a model directory
PACK_JSON_FILES = {
    "audio8_config": "config.json",
    "audio8_generation_config": "generation_config.json",
    "audio8_processor_config": "processor_config.json",
    "audio8_tokenizer_config": "tokenizer_config.json",
    "audio8_special_tokens_map": "special_tokens_map.json",
    "audio8_tokenizer_json": "tokenizer.json",
}
PACK_REQUIRED_JSON = ("audio8_config", "audio8_tokenizer_config",
                      "audio8_special_tokens_map", "audio8_tokenizer_json")

ARCH_FILE_NAMES = ("configuration_arktts.py", "modeling_arktts.py",
                   "modeling_arktts_codec.py", "processing_arktts.py")


def is_pack_path(path) -> bool:
    return Path(path).suffix.lower() == PACK_SUFFIX


def pack_model_directory(model_dir, target, overwrite=False, quantize=PACK_EXTENSION) -> dict:
    """Pack a model directory into one .safetensors file.

    The source directory stays untouched. The codec state dict is stored
    normalized exactly the way ArkttsModel.load_codec() consumes it.
    ``quantize`` is "none" or "int8_convrot"; the int8 variant quantises the
    eligible nn.Linear weights via the bundled architecture and records
    audio8_quant=int8_convrot plus the audio8_int8 parameter map.
    """

    model_dir = Path(model_dir)
    if quantize not in PACK_QUANTIZATIONS:
        raise ValueError(
            f"[Audio8] Unknown pack quantize mode {quantize!r}; "
            f"expected one of {PACK_QUANTIZATIONS}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"[Audio8] Model directory not found: {model_dir}")
    for name in ("config.json", "codec.pth", "model.safetensors"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(
                f"[Audio8] Cannot pack {model_dir}: missing {name}")

    target = Path(target)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"[Audio8] Pack already exists: {target} "
            "(set overwrite to true to replace it)")

    tensors = {}
    for key, tensor in safetensors_load_file(str(model_dir / "model.safetensors")).items():
        tensors[MODEL_PREFIX + key] = tensor

    codec_state = torch.load(str(model_dir / "codec.pth"), map_location="cpu",
                             weights_only=True)
    if isinstance(codec_state, dict) and "state_dict" in codec_state:
        codec_state = codec_state["state_dict"]
    if any("generator." in key for key in codec_state):
        codec_state = {
            key.replace("generator.", ""): value
            for key, value in codec_state.items() if "generator." in key
        }
    codec_state = {
        key: value
        for key, value in codec_state.items()
        if not key.endswith(("freqs_cis", "causal_mask"))
    }
    for key, tensor in codec_state.items():
        tensors[CODEC_PREFIX + key] = tensor

    metadata = {
        "audio8_format": PACK_FORMAT,
        "audio8_quant": PACK_EXTENSION,
        "audio8_model_id": MODEL_ID,
        "audio8_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for meta_key, file_name in PACK_JSON_FILES.items():
        source = model_dir / file_name
        if source.is_file():
            metadata[meta_key] = source.read_text(encoding="utf-8")
    missing_json = [key for key in PACK_REQUIRED_JSON if key not in metadata]
    if missing_json:
        raise FileNotFoundError(
            f"[Audio8] Cannot pack {model_dir}: missing "
            + ", ".join(PACK_JSON_FILES[key] for key in missing_json))

    source_arch = {}
    for name in ARCH_FILE_NAMES:
        source = model_dir / name
        if source.is_file():
            source_arch[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_arch:
        metadata["audio8_source_arch_sha"] = json.dumps(source_arch)

    quantized_count = 0
    if quantize == INT8_CONVROT:
        quantized_count = _apply_int8_convrot_to_tensors(model_dir, tensors, metadata)
        metadata["audio8_quant"] = INT8_CONVROT

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        safetensors_save_file(tensors, str(tmp), metadata=metadata)
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    info = {
        "path": str(target),
        "model_tensors": sum(1 for key in tensors if key.startswith(MODEL_PREFIX)),
        "codec_tensors": sum(1 for key in tensors if key.startswith(CODEC_PREFIX)),
        "quantized": quantized_count,
        "size_bytes": target.stat().st_size,
        "source_arch_sha": source_arch,
    }
    logger.info("[Audio8] Packed %s -> %s (%d model + %d codec tensors, %d int8, %.2f GiB)",
                model_dir, target, info["model_tensors"], info["codec_tensors"],
                quantized_count, info["size_bytes"] / 2**30)
    return info


def _apply_int8_convrot_to_tensors(model_dir, tensors, metadata) -> int:
    """Quantise the model.* linear weights inside ``tensors`` in place.

    The bundled architecture materialises the directory's float weights so
    the eligible-module rule matches the runtime exactly, quantises them on
    the accelerator when available (the same device the ComfyUI runtime uses
    for on-the-fly int8_convrot, which keeps both paths comparable) and
    replaces each float entry with the packed int8 tensor plus a
    "<path>.weight_scale" float32 companion.
    """

    from .audio8_arch.configuration_arktts import ArkttsConfig
    from .audio8_arch.modeling_arktts import ArkttsModel
    from .audio8_quantization import (collect_int8_convrot_state,
                                      quantize_model_int8_convrot)

    config = ArkttsConfig(**json.loads(metadata["audio8_config"]))
    model = ArkttsModel(config)
    # The ComfyUI runtime quantises after model.to(bfloat16); materialise in
    # the same dtype so both paths see identical input bits and therefore
    # produce identical packed tensors.
    model.to(dtype=torch.bfloat16)
    float_state = {key[len(MODEL_PREFIX):]: value
                   for key, value in tensors.items() if key.startswith(MODEL_PREFIX)}
    missing, unexpected = model.load_state_dict(float_state, strict=False)
    buffer_names = {name for name, _ in model.named_buffers()}
    real_missing = [name for name in missing if name not in buffer_names]
    if unexpected or real_missing:
        raise ValueError(
            f"[Audio8] Cannot materialise {model_dir} for quantisation: "
            f"missing={real_missing[:8]} unexpected={list(unexpected)[:8]}")
    del float_state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    count = quantize_model_int8_convrot(model, INT8_CONVROT_GROUP_SIZE)
    if not count:
        raise ValueError("[Audio8] No quantisable nn.Linear weights found")
    quantized_tensors, int8_meta = collect_int8_convrot_state(model)
    del model

    for path, parts in quantized_tensors.items():
        key = MODEL_PREFIX + path + ".weight"
        if key not in tensors:
            raise ValueError(f"[Audio8] Quantised module {path} has no source weight")
        tensors[key] = parts[""]
        tensors[MODEL_PREFIX + path + INT8_SCALE_SUFFIX] = parts["_scale"]
    metadata["audio8_int8"] = json.dumps(int8_meta)
    logger.info("[Audio8] Pack quantised %d nn.Linear weights to int8_convrot "
                "(group %d) on %s", count, INT8_CONVROT_GROUP_SIZE, device)
    return count


def _split_int8_pack_state(pack_path, model_state, int8_meta) -> tuple:
    """Split an int8_convrot pack state into (float state, layout entries).

    Quantised modules keep the plain "<path>.weight" key as an int8 tensor
    plus a "<path>.weight_scale" float32 companion; every entry must be
    listed in the pack's audio8_int8 metadata map and vice versa.
    """

    entries = {}
    float_state = {}
    for key, value in model_state.items():
        if key.endswith(INT8_SCALE_SUFFIX):
            continue  # consumed with its .weight companion below
        if key.endswith(".weight") and value.dtype == torch.int8:
            path = key[:-len(".weight")]
            scale_key = path + INT8_SCALE_SUFFIX
            if scale_key not in model_state:
                raise ValueError(
                    f"[Audio8] Pack {pack_path}: int8 weight {key} has no "
                    f"{INT8_SCALE_SUFFIX} companion")
            if path not in int8_meta:
                raise ValueError(
                    f"[Audio8] Pack {pack_path}: no audio8_int8 entry for {path}")
            entries[path] = {"": value, "_scale": model_state[scale_key]}
        else:
            float_state[key] = value
    unlisted = set(int8_meta) - set(entries)
    if unlisted:
        raise ValueError(
            f"[Audio8] Pack {pack_path}: audio8_int8 lists "
            f"{sorted(unlisted)[:8]} without an int8 weight tensor")
    return float_state, entries


def read_pack_metadata(pack_path) -> dict:
    """Return the metadata of an Audio8 pack; raise unless it is a valid pack."""

    pack_path = Path(pack_path)
    if not pack_path.is_file():
        raise FileNotFoundError(f"[Audio8] Pack file not found: {pack_path}")
    with safe_open(str(pack_path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("audio8_format") != PACK_FORMAT:
        raise ValueError(
            f"[Audio8] {pack_path} is not an Audio8 pack v{PACK_FORMAT} "
            f"(audio8_format={metadata.get('audio8_format')!r}). Use the "
            "Audio8 Pack Model node to create it from a model directory.")
    if metadata.get("audio8_quant", PACK_EXTENSION) not in PACK_QUANTIZATIONS:
        raise NotImplementedError(
            f"[Audio8] Pack {pack_path} declares quantization "
            f"{metadata['audio8_quant']!r}; this node understands "
            f"{PACK_EXTENSION!r} and {INT8_CONVROT!r} only. Quantise a "
            "directory pack with the Audio8 Pack Model node instead.")
    if metadata.get("audio8_quant") == INT8_CONVROT:
        try:
            int8_meta = json.loads(metadata.get("audio8_int8") or "")
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[Audio8] Pack {pack_path} has no valid audio8_int8 metadata") from exc
        if not isinstance(int8_meta, dict) or not int8_meta:
            raise ValueError(
                f"[Audio8] Pack {pack_path} declares {INT8_CONVROT} but carries "
                "no audio8_int8 parameter map")
    missing_json = [key for key in PACK_REQUIRED_JSON if key not in metadata]
    if missing_json:
        raise ValueError(
            f"[Audio8] Pack {pack_path} is missing metadata: "
            + ", ".join(missing_json))
    return metadata


def build_pack_tokenizer(metadata):
    """Rebuild the fast tokenizer from pack metadata without any files.

    Lazy imports: unit tests stub `transformers`, and importing the real
    HF stack is only needed when a tokenizer is actually built.
    """

    from tokenizers import Tokenizer
    from transformers import AddedToken, PreTrainedTokenizerFast

    def materialize(value):
        # special_tokens_map.json serializes AddedTokens as dicts
        # (content/lstrip/rstrip/normalized/single_word); HF's
        # from_pretrained converts them before passing init kwargs.
        if isinstance(value, dict) and "content" in value:
            return AddedToken(**value)
        if isinstance(value, list):
            return [materialize(item) for item in value]
        return value

    tokenizer_object = Tokenizer.from_str(metadata["audio8_tokenizer_json"])
    tokenizer_config = json.loads(metadata["audio8_tokenizer_config"])
    special_tokens_map = json.loads(metadata["audio8_special_tokens_map"])
    # tokenizer_class and added_tokens_decoder are handled by the tokenizer
    # itself (tokenizer.json carries the added tokens); everything else are
    # plain HF init kwargs. special_tokens_map.json takes precedence, as in
    # AutoTokenizer.from_pretrained.
    init_kwargs = {}
    for key, value in tokenizer_config.items():
        if key in ("tokenizer_class", "added_tokens_decoder"):
            continue
        init_kwargs[key] = value
    init_kwargs.update(special_tokens_map)
    init_kwargs = {key: materialize(value) for key, value in init_kwargs.items()}
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer_object, **init_kwargs)


def load_pack_model(pack_path, compute_dtype, device) -> tuple:
    """Load (model, processor) from a pack using the bundled architecture.

    No transformers loading machinery is involved: the config object, both
    modules and the tokenizer are constructed from pack metadata and the
    node's audited Apache-2.0 architecture copy, then weights are applied
    via load_state_dict. The codec is injected into the model's lazy
    load_codec slot so no codec.pth file lookup can happen.
    """

    pack_path = Path(pack_path)
    metadata = read_pack_metadata(pack_path)
    from .audio8_arch.configuration_arktts import ArkttsConfig
    from .audio8_arch.modeling_arktts import ArkttsModel, _precompute_rope
    from .audio8_arch.modeling_arktts_codec import ArkttsCodec
    from .audio8_arch.processing_arktts import ArkttsProcessor
    config = ArkttsConfig(**json.loads(metadata["audio8_config"]))
    model = ArkttsModel(config)

    tensors = safetensors_load_file(str(pack_path))
    model_state = {key[len(MODEL_PREFIX):]: value
                   for key, value in tensors.items() if key.startswith(MODEL_PREFIX)}
    codec_state = {key[len(CODEC_PREFIX):]: value
                   for key, value in tensors.items() if key.startswith(CODEC_PREFIX)}
    if not model_state or not codec_state:
        raise ValueError(f"[Audio8] Pack {pack_path} has no "
                         f"{MODEL_PREFIX}* or {CODEC_PREFIX}* tensors")

    # int8_convrot packs store eligible nn.Linear weights as int8 with a
    # float32 scale companion; those modules get weight=None before the
    # float state dict is applied and the layout is installed afterwards.
    int8_meta = {}
    int8_entries = {}
    if metadata.get("audio8_quant") == INT8_CONVROT:
        int8_meta = json.loads(metadata["audio8_int8"])
        model_state, int8_entries = _split_int8_pack_state(pack_path, model_state, int8_meta)
        for path in int8_entries:
            module = model.get_submodule(path)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"[Audio8] Pack int8 target {path} is no nn.Linear")
            module.weight = None

    # strict=False, then exact-set validation: only registered buffers that
    # the checkpoint legitimately omits (recomputed after load) may be missing.
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    buffer_names = {name for name, _ in model.named_buffers()}
    real_missing = [name for name in missing if name not in buffer_names]
    if unexpected or real_missing or missing:
        raise RuntimeError(
            f"[Audio8] Pack state dict mismatch for {pack_path}: "
            f"unexpected={unexpected[:8]} missing={missing[:8]} "
            f"(missing outside known buffers: {real_missing[:8]}). The pack "
            "does not match the node's bundled architecture.")

    if int8_entries:
        from .audio8_quantization import apply_int8_convrot_state
        installed = apply_int8_convrot_state(model, int8_entries, int8_meta, device,
                                             dtype=compute_dtype)
        if installed != len(int8_meta):
            raise RuntimeError(
                f"[Audio8] Pack {pack_path}: installed {installed} of "
                f"{len(int8_meta)} int8 modules")

    # Transformers 5 materializes non-persistent buffers without their
    # constructor values; this explicit path recomputes them unconditionally.
    model.freqs_cis = _precompute_rope(config.max_seq_len, config.head_dim, config.rope_base)
    model.fast_freqs_cis = _precompute_rope(config.num_codebooks, config.fast_head_dim, config.rope_base)

    codec = ArkttsCodec(config)
    codec.load_state_dict(codec_state, strict=True)
    codec.eval()
    # Same slot ArkttsModel.load_codec() uses; injecting it prevents any
    # codec.pth file resolution against _name_or_path.
    model.__dict__["_arktts_codec"] = codec

    model.eval()

    tokenizer = build_pack_tokenizer(metadata)
    processor_config = json.loads(metadata.get("audio8_processor_config", "{}"))
    processor = ArkttsProcessor(
        tokenizer=tokenizer,
        num_codebooks=processor_config.get("num_codebooks", 10),
        semantic_begin_id=processor_config.get("semantic_begin_id", 151678),
        audio_sampling_rate=processor_config.get("audio_sampling_rate", 44100),
    )
    return model, processor
