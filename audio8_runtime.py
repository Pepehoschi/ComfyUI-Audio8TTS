import logging
from contextlib import ExitStack
from importlib.metadata import version

import torch
from packaging.version import Version

import comfy.model_management
import comfy.audio
import comfy.utils

from .audio8_paths import resolve_pack
from .audio8_quantization import INT8_CONVROT, quantize_model_int8_convrot
from .audio8_acceleration import (GRAPH_SCOPES, cuda_graph_fast_decode,
                                   cuda_graph_frame_decode)
from .audio8_comfy import comfy_audio8_ops
from .audio8_chunking import split_text
from .audio8_script import plan_script_chunks
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


# Packs are the only loadable format: the selection arrives as a name from
# the host's diffusion_models folders (audio8_paths.resolve_pack) and the
# node's bundled Apache-2.0 architecture (audio8_arch) is instantiated
# directly - no remote code, no trust_remote_code, no Auto* registration.
def load_model(dtype="auto", ckpt_name=None):
    check_transformers_version()
    pack_path = resolve_pack(ckpt_name)
    device = comfy.model_management.get_torch_device()
    dtypes = {"auto": torch.bfloat16, "bfloat16": torch.bfloat16,
              "float16": torch.float16, "float32": torch.float32}
    if dtype not in dtypes and dtype != INT8_CONVROT:
        raise ValueError(f"[Audio8] Unknown dtype: {dtype}")
    compute_dtype = torch.float32 if device.type == "cpu" else dtypes.get(dtype, torch.bfloat16)
    return _load_pack_bundle(pack_path, dtype, compute_dtype, device)


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
    # int8_convrot packs arrive with their layouts already installed by
    # audio8_pack; re-quantising them would be a dtype-on-dtype no-op at best.
    quantization = INT8_CONVROT if getattr(model, "audio8_int8_quantised", False) else "none"
    if dtype == INT8_CONVROT and quantization != INT8_CONVROT:
        quantize_model_int8_convrot(model)
        quantization = INT8_CONVROT
    elif dtype == INT8_CONVROT and quantization == INT8_CONVROT:
        logger.info("[Audio8] Pack %s is already int8_convrot; dtype needs no "
                    "on-the-fly quantisation", pack_path)
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
             comfy_ops=False, attention_backend="auto", cuda_graph_scope="fast", script=None):
    if cuda_graph_scope not in GRAPH_SCOPES:
        raise ValueError(f"[Audio8] Unknown cuda_graph_scope: {cuda_graph_scope!r}")
    planned = None
    if script is not None:
        planned = plan_script_chunks(script, chunking, target_chunk_chars,
                                     max_chunk_chars, sentence_pause_ms,
                                     paragraph_pause_ms)
        if not planned:
            raise ValueError("[Audio8] Script contains no renderable text")
    elif not text.strip():
        raise ValueError("[Audio8] Text must not be empty")
    cuda_graphs = cuda_graphs and bundle.device.type == "cuda"
    generator = torch.Generator(device=bundle.device).manual_seed(seed)
    settings = dict(temperature=temperature, top_p=top_p, top_k=top_k,
                    do_sample=not greedy, generator=generator, return_dict_in_generate=True)
    if planned is not None:
        parts = []
        progress = comfy.utils.ProgressBar(len(planned))
        for index, (chunk_text, chunk_reference, pause_after_ms) in enumerate(planned, 1):
            comfy.model_management.throw_exception_if_processing_interrupted()
            logger.info("[Audio8] Script chunk %d/%d", index, len(planned))
            try:
                part = _generate_utterance(bundle, chunk_text,
                                          chunk_reference if chunk_reference is not None else reference,
                                          max_new_tokens, retry_max_new_tokens, settings,
                                          fade_out_ms, cuda_graphs, comfy_ops,
                                          attention_backend, cuda_graph_scope)
            except Exception as exc:
                raise RuntimeError(
                    f"[Audio8] Generation failed for script chunk {index}/{len(planned)}: "
                    f"{chunk_text[:80]!r}") from exc
            parts.append(part)
            if pause_after_ms > 0:
                samples = round(bundle.sample_rate * pause_after_ms / 1000)
                parts.append(part.new_zeros((1, 1, samples)))
            progress.update(1)
        waveform = torch.cat(parts, dim=-1)
    elif chunking == "disabled" or len(text) <= max_chunk_chars or not text.strip():
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
