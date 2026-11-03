import logging
from contextlib import ExitStack
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import torch
from packaging.version import Version
from transformers import AutoModel, AutoProcessor

import comfy.model_management
import comfy.audio
import comfy.utils

from .audio8_paths import ARCH_MIRROR_SUFFIX, local_model_path
from .audio8_quantization import INT8_CONVROT, quantize_model_int8_convrot
from .audio8_acceleration import (GRAPH_SCOPES, cuda_graph_fast_decode,
                                   cuda_graph_frame_decode)
from .audio8_comfy import comfy_audio8_ops
from .audio8_chunking import split_text
from .audio8_types import (Audio8ModelBundle, Audio8Reference, MODEL_ID,
                           CODEC_SAMPLE_RATE, NUM_CODEBOOKS, CODEBOOK_SIZE,
                           validate_reference)


logger = logging.getLogger(__name__)

# Constants the nodes were written and validated against; the checkpoint config
# is compared with them at load so an incompatible model fails loudly instead
# of producing subtly wrong audio.
EXPECTED_CONFIG = {
    "model_type": "arktts",
    "codec_sample_rate": CODEC_SAMPLE_RATE,
    "num_codebooks": NUM_CODEBOOKS,
    "codebook_size": CODEBOOK_SIZE,
}


def validate_model_config(config, model_path):
    """Abort when the checkpoint config contradicts the integration constants."""

    mismatches = [
        f"{key}: expected {expected!r}, config reports {getattr(config, key, None)!r}"
        for key, expected in EXPECTED_CONFIG.items()
        if getattr(config, key, None) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "[Audio8] Model incompatible with this integration ("
            + "; ".join(mismatches) + f"). Checkpoint directory: {model_path}")


def check_transformers_version():
    installed = Version(version("transformers"))
    if not Version("4.57.0") <= installed < Version("6"):
        raise RuntimeError(
            f"[Audio8] Transformers {installed} is installed; this integration currently supports "
            ">=4.57.0,<6. Use a compatible ComfyUI environment. No packages have been changed."
        )


# Architecture files the model repos ship; if they are absent, the node builds
# a mirror of the model directory (hard links) containing verbatim copies of
# its bundled Apache-2.0 copy (audio8_arch) and loads through that mirror via
# the proven dynamic-module path.
ARCH_CODE_FILES = ("configuration_arktts.py", "modeling_arktts.py",
                   "modeling_arktts_codec.py", "processing_arktts.py")


def build_arch_mirror(model_path):
    """Mirror a model directory and add the node's bundled architecture files.

    Model directories without the four Arktts architecture .py files are loaded
    through a sibling mirror directory (<name>__with_arch) whose entries are
    hard links to the originals (copies across volumes), plus verbatim copies
    of the bundled Apache-2.0 architecture (audio8_arch/). transformers then
    resolves them via its proven dynamic-module path; no Auto* registration is
    used because transformers 5.x silently leaves registered custom classes
    initialized with constructor values instead of checkpoint weights.
    The mirror doubles as a cache and is rebuilt when model files change.
    """

    import hashlib
    import os
    import shutil

    model_path = Path(model_path)
    mirror = model_path.parent / (model_path.name + ARCH_MIRROR_SUFFIX)

    fingerprint_src = hashlib.sha256()
    for entry in sorted(model_path.iterdir(), key=lambda p: p.name):
        if entry.is_file():
            stat = entry.stat()
            fingerprint_src.update(f"{entry.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    fingerprint = fingerprint_src.hexdigest()

    stamp = mirror / ".audio8_mirror_fingerprint"
    if mirror.is_dir() and stamp.is_file() and stamp.read_text(encoding="utf-8") == fingerprint:
        return mirror

    from . import audio8_arch
    arch_sources = [Path(audio8_arch.__file__).parent / name
                    for name in ("configuration_arktts.py", "modeling_arktts.py",
                                 "modeling_arktts_codec.py", "processing_arktts.py")]
    for source in arch_sources:
        if not source.is_file():
            raise FileNotFoundError(f"[Audio8] bundled architecture file missing: {source}")

    if mirror.is_dir():
        shutil.rmtree(mirror)
    mirror.mkdir()
    try:
        for entry in sorted(model_path.iterdir(), key=lambda p: p.name):
            if entry.is_file():
                target = mirror / entry.name
                try:
                    os.link(entry, target)
                except OSError:
                    shutil.copy2(entry, target)
        for source in arch_sources:
            shutil.copy2(source, mirror / source.name)
        stamp.write_text(fingerprint, encoding="utf-8")
    except Exception:
        shutil.rmtree(mirror, ignore_errors=True)
        raise
    return mirror


def load_model(dtype="auto", model_name="auto", model_path=""):
    check_transformers_version()
    model_path = local_model_path(model_name, model_path)
    device = comfy.model_management.get_torch_device()
    dtypes = {"auto": torch.bfloat16, "bfloat16": torch.bfloat16,
              "float16": torch.float16, "float32": torch.float32}
    if dtype not in dtypes and dtype != INT8_CONVROT:
        raise ValueError(f"[Audio8] Unknown dtype: {dtype}")
    compute_dtype = torch.float32 if device.type == "cpu" else dtypes.get(dtype, torch.bfloat16)
    if model_path.suffix.lower() == ".safetensors":
        return _load_pack_bundle(model_path, dtype, compute_dtype, device)
    if not all((model_path / name).is_file() for name in ARCH_CODE_FILES):
        mirror = build_arch_mirror(model_path)
        logger.info("[Audio8] %s ships no architecture files; loading through the "
                    "mirror %s with the node's bundled Apache-2.0 architecture "
                    "(hard links to the original weights; audit copies in audio8_arch/)",
                    model_path, mirror)
        model_path = mirror
    logger.info("[Audio8] Loading %s from %s as %s on %s", MODEL_ID, model_path, compute_dtype, device)
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True, dtype=compute_dtype,
    )
    # Transformers 5 materializes non-persistent buffers without their constructor values.
    precompute_rope = import_module(type(model).__module__)._precompute_rope
    config = model.config
    validate_model_config(config, model_path)
    model.freqs_cis = precompute_rope(config.max_seq_len, config.head_dim, config.rope_base)
    model.fast_freqs_cis = precompute_rope(config.num_codebooks, config.fast_head_dim, config.rope_base)
    model.eval()
    model.to(device)
    quantization = "none"
    if dtype == INT8_CONVROT:
        quantize_model_int8_convrot(model)
        quantization = INT8_CONVROT
    return Audio8ModelBundle(model, processor, MODEL_ID, device, compute_dtype,
                             int(model.config.codec_sample_rate), quantization=quantization)


def _load_pack_bundle(pack_path, dtype, compute_dtype, device):
    """Load a single-file pack via the node's bundled architecture.

    No trust_remote_code and no Auto* registration: audio8_pack builds the
    config, both modules and the tokenizer from pack metadata plus the
    audited Apache-2.0 copies in audio8_arch, and applies weights through
    load_state_dict with an exact key-set check.
    """

    from .audio8_pack import load_pack_model

    logger.info("[Audio8] Loading %s pack %s as %s on %s",
                MODEL_ID, pack_path, compute_dtype, device)
    model, processor = load_pack_model(pack_path, compute_dtype, device)
    validate_model_config(model.config, pack_path)
    model.to(device=device, dtype=compute_dtype)
    quantization = "none"
    if dtype == INT8_CONVROT:
        quantize_model_int8_convrot(model)
        quantization = INT8_CONVROT
    return Audio8ModelBundle(model, processor, MODEL_ID, device, compute_dtype,
                             int(model.config.codec_sample_rate), quantization=quantization)


def reference_audio_values(audio, sample_rate):
    waveform = audio["waveform"]
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("[Audio8] Reference audio must have shape [1, channels, samples]")
    waveform = waveform.mean(dim=1, keepdim=True)
    return comfy.audio.resample(waveform, audio["sample_rate"], sample_rate)


def encode_reference(bundle, audio, reference_text):
    values = reference_audio_values(audio, bundle.sample_rate).to(bundle.device)
    lengths = torch.tensor([values.shape[-1]], dtype=torch.long, device=bundle.device)
    # PyTorch 2.8.0+cu128 changes BF16 Snake arithmetic after TorchScript warmup.
    with torch.jit.optimized_execution(False):
        codes, code_lengths = bundle.model.encode_audio(values, lengths)
    valid_codes = codes[0, :, :int(code_lengths[0])]
    reference = Audio8Reference(
        codes=valid_codes.detach().to(device="cpu", dtype=torch.long).contiguous(),
        reference_text=reference_text, model_id=bundle.model_id,
        codec_sample_rate=bundle.sample_rate,
    )
    return validate_reference(reference)


def _model_generate(bundle, inputs, max_new_tokens, settings, cuda_graphs, comfy_ops, attention_backend="auto",
                    cuda_graph_scope="fast"):
    if comfy_ops and getattr(bundle.model, "audio8_int8_quantised", False):
        logger.warning(
            "[Audio8] Disabling comfy_ops: it re-patches the Linear layers and is incompatible "
            "with int8_convrot weights.")
        comfy_ops = False
    with ExitStack() as stack:
        if comfy_ops:
            stack.enter_context(comfy_audio8_ops(bundle, attention_backend=attention_backend))
        if cuda_graphs:
            if cuda_graph_scope == "frame":
                stack.enter_context(cuda_graph_frame_decode(bundle, max_new_tokens))
            else:
                stack.enter_context(cuda_graph_fast_decode(bundle))
        return bundle.model.generate(**inputs, max_new_tokens=max_new_tokens, **settings)


def _effective_retry_budget(max_seq_len, prompt_tokens, requested):
    """Upstream caps any generate request to max_seq_len minus the prompt length."""

    if max_seq_len <= 0 or prompt_tokens <= 0:
        return requested
    return min(requested, max_seq_len - prompt_tokens)


def _generate_utterance(bundle, text, reference, max_new_tokens, retry_max_new_tokens, settings, fade_out_ms,
                        cuda_graphs, comfy_ops, attention_backend, cuda_graph_scope="fast"):
    kwargs = {}
    if reference is not None:
        validate_reference(reference)
        kwargs = {"reference_codes": reference.codes,
                  "reference_text": [reference.reference_text]}
    inputs = bundle.processor(text=[text], return_tensors="pt", **kwargs)
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
    output = _model_generate(bundle, inputs, max_new_tokens, settings, cuda_graphs, comfy_ops, attention_backend,
                             cuda_graph_scope)
    if not bool(output.finished[0]) and retry_max_new_tokens > max_new_tokens:
        max_seq_len = int(getattr(getattr(bundle.model, "config", None), "max_seq_len", 0) or 0)
        prompt_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        retry_budget = _effective_retry_budget(max_seq_len, prompt_tokens, retry_max_new_tokens)
        if retry_budget != retry_max_new_tokens:
            logger.info("[Audio8] Retry budget capped from %d to %d tokens "
                        "(max_seq_len %d minus prompt %d)", retry_max_new_tokens,
                        retry_budget, max_seq_len, prompt_tokens)
        if retry_budget > max_new_tokens:
            logger.info("[Audio8] No EOS; retrying with %d tokens", retry_budget)
            output = _model_generate(bundle, inputs, retry_budget, settings, cuda_graphs, comfy_ops,
                                     attention_backend, cuda_graph_scope)
    length = int(output.code_lengths[0])
    if length <= 0:
        raise RuntimeError("[Audio8] Generation produced no valid codec frames")
    if not bool(output.finished[0]):
        logger.warning("[Audio8] Generation reached the token limit without EOS")
    with torch.jit.optimized_execution(False):
        waveforms, waveform_lengths = bundle.model.decode_audio(output.codes[:, :, :length])
    samples = int(waveform_lengths[0])
    if samples <= 0:
        raise RuntimeError("[Audio8] Codec produced no audio samples")
    waveform = waveforms[0, :samples].detach().float().cpu().reshape(1, 1, -1)
    fade_samples = min(samples, max(0, round(bundle.sample_rate * fade_out_ms / 1000)))
    if fade_samples:
        waveform = waveform.clone()
    if fade_samples > 1:
        waveform[..., -fade_samples:] *= torch.linspace(
            1.0, 0.0, fade_samples, device=waveform.device, dtype=waveform.dtype)
    elif fade_samples == 1:
        waveform[..., -1] = 0
    return waveform


def generate(bundle, text, reference=None, seed=42, max_new_tokens=1024,
             temperature=0.8, top_p=0.95, top_k=50, greedy=False, retry_max_new_tokens=2000,
             chunking="auto", target_chunk_chars=130, max_chunk_chars=150,
             sentence_pause_ms=180, paragraph_pause_ms=350, fade_out_ms=5.0, cuda_graphs=False,
             comfy_ops=False, attention_backend="auto", cuda_graph_scope="fast"):
    if cuda_graph_scope not in GRAPH_SCOPES:
        raise ValueError(f"[Audio8] Unknown cuda_graph_scope: {cuda_graph_scope!r}")
    if not text.strip():
        raise ValueError("[Audio8] Text must not be empty")
    cuda_graphs = cuda_graphs and bundle.device.type == "cuda"
    generator = torch.Generator(device=bundle.device).manual_seed(seed)
    settings = dict(temperature=temperature, top_p=top_p, top_k=top_k,
                    do_sample=not greedy, generator=generator, return_dict_in_generate=True)
    if chunking == "disabled" or len(text) <= max_chunk_chars or not text.strip():
        waveform = _generate_utterance(bundle, text, reference, max_new_tokens,
                                       retry_max_new_tokens, settings, fade_out_ms, cuda_graphs, comfy_ops,
                                       attention_backend, cuda_graph_scope)
    else:
        chunks = split_text(text, target_chunk_chars, max_chunk_chars,
                            sentence_pause_ms, paragraph_pause_ms)
        parts = []
        progress = comfy.utils.ProgressBar(len(chunks))
        for index, chunk in enumerate(chunks, 1):
            comfy.model_management.throw_exception_if_processing_interrupted()
            logger.info("[Audio8] Chunk %d/%d", index, len(chunks))
            try:
                part = _generate_utterance(bundle, chunk.text, reference, max_new_tokens,
                                          retry_max_new_tokens, settings, fade_out_ms, cuda_graphs, comfy_ops,
                                          attention_backend, cuda_graph_scope)
            except Exception as exc:
                raise RuntimeError(
                    f"[Audio8] Generation failed for chunk {index}/{len(chunks)}: {chunk.text[:80]!r}"
                ) from exc
            parts.append(part)
            if chunk.pause_after_ms > 0:
                samples = round(bundle.sample_rate * chunk.pause_after_ms / 1000)
                parts.append(part.new_zeros((1, 1, samples)))
            progress.update(1)
        waveform = torch.cat(parts, dim=-1)
    return {"waveform": waveform, "sample_rate": bundle.sample_rate}
