import logging
from contextlib import ExitStack
from importlib import import_module
from importlib.metadata import version

import torch
from packaging.version import Version
from transformers import AutoModel, AutoProcessor

import comfy.model_management
import comfy.audio
import comfy.utils

from .audio8_paths import local_model_path
from .audio8_acceleration import cuda_graph_fast_decode
from .audio8_comfy import comfy_audio8_ops
from .audio8_chunking import split_text
from .audio8_types import Audio8ModelBundle, Audio8Reference, MODEL_ID, validate_reference


logger = logging.getLogger(__name__)


def check_transformers_version():
    installed = Version(version("transformers"))
    if not Version("4.57.0") <= installed < Version("6"):
        raise RuntimeError(
            f"[Audio8] Transformers {installed} is installed; this integration currently supports "
            ">=4.57.0,<6. Use a compatible ComfyUI environment. No packages have been changed."
        )


def load_model(dtype="auto"):
    check_transformers_version()
    model_path = local_model_path()
    device = comfy.model_management.get_torch_device()
    dtypes = {"auto": torch.bfloat16, "bfloat16": torch.bfloat16,
              "float16": torch.float16, "float32": torch.float32}
    if dtype not in dtypes:
        raise ValueError(f"[Audio8] Unknown dtype: {dtype}")
    compute_dtype = torch.float32 if device.type == "cpu" else dtypes[dtype]
    logger.info("[Audio8] Loading %s as %s on %s", MODEL_ID, compute_dtype, device)
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True, dtype=compute_dtype,
    )
    # Transformers 5 materializes non-persistent buffers without their constructor values.
    precompute_rope = import_module(type(model).__module__)._precompute_rope
    config = model.config
    model.freqs_cis = precompute_rope(config.max_seq_len, config.head_dim, config.rope_base)
    model.fast_freqs_cis = precompute_rope(config.num_codebooks, config.fast_head_dim, config.rope_base)
    model.eval()
    model.to(device)
    return Audio8ModelBundle(model, processor, MODEL_ID, device, compute_dtype,
                             int(model.config.codec_sample_rate))


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


def _model_generate(bundle, inputs, max_new_tokens, settings, cuda_graphs, comfy_ops, attention_backend="auto"):
    with ExitStack() as stack:
        if comfy_ops:
            stack.enter_context(comfy_audio8_ops(bundle, attention_backend=attention_backend))
        if cuda_graphs:
            stack.enter_context(cuda_graph_fast_decode(bundle))
        return bundle.model.generate(**inputs, max_new_tokens=max_new_tokens, **settings)


def _generate_utterance(bundle, text, reference, max_new_tokens, retry_max_new_tokens, settings, fade_out_ms,
                        cuda_graphs, comfy_ops, attention_backend):
    kwargs = {}
    if reference is not None:
        validate_reference(reference)
        kwargs = {"reference_codes": reference.codes,
                  "reference_text": [reference.reference_text]}
    inputs = bundle.processor(text=[text], return_tensors="pt", **kwargs)
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
    output = _model_generate(bundle, inputs, max_new_tokens, settings, cuda_graphs, comfy_ops, attention_backend)
    if not bool(output.finished[0]) and retry_max_new_tokens > max_new_tokens:
        logger.info("[Audio8] No EOS; retrying with %d tokens", retry_max_new_tokens)
        output = _model_generate(bundle, inputs, retry_max_new_tokens, settings, cuda_graphs, comfy_ops,
                                 attention_backend)
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
             comfy_ops=False, attention_backend="auto"):
    cuda_graphs = cuda_graphs and bundle.device.type == "cuda"
    generator = torch.Generator(device=bundle.device).manual_seed(seed)
    settings = dict(temperature=temperature, top_p=top_p, top_k=top_k,
                    do_sample=not greedy, generator=generator, return_dict_in_generate=True)
    if chunking == "disabled" or len(text) <= max_chunk_chars or not text.strip():
        waveform = _generate_utterance(bundle, text, reference, max_new_tokens,
                                       retry_max_new_tokens, settings, fade_out_ms, cuda_graphs, comfy_ops,
                                       attention_backend)
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
                                          attention_backend)
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
