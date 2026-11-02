"""Temporary ComfyUI operation adapter for the Audio8 Transformers model."""

from contextlib import contextmanager

import torch

import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import (
    AttentionTensorContainer, ComfyAttention, get_attention_function, optimized_attention,
)


operations = comfy.ops.disable_weight_init


def _rope_matrix(rope: torch.Tensor) -> torch.Tensor:
    """Convert Audio8's complex-pair RoPE buffer to ComfyUI's rotation matrix."""

    values = rope.float()
    cosine = values[..., 0]
    sine = values[..., 1]
    matrix = torch.stack((cosine, -sine, sine, cosine), dim=-1)
    matrix = matrix.reshape(*matrix.shape[:-1], 2, 2)
    if matrix.ndim == 4:
        return matrix.unsqueeze(0).unsqueeze(0)
    if matrix.ndim == 5:
        return matrix.unsqueeze(1)
    raise ValueError(f"Unexpected Audio8 RoPE shape: {tuple(rope.shape)}")


def _native_attention(module, x, rope, attention_mask, cache_position=None):
    batch, length, _ = x.shape
    query_size = module.n_head * module.head_dim
    kv_size = module.n_local_heads * module.head_dim
    query, key, value = module.wqkv(x).split((query_size, kv_size, kv_size), dim=-1)
    query = query.view(batch, length, module.n_head, module.head_dim)
    key = key.view(batch, length, module.n_local_heads, module.head_dim)
    value = value.view(batch, length, module.n_local_heads, module.head_dim)
    if module.qk_norm:
        query = module.q_norm(query)
        key = module.k_norm(key)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query, key = apply_rope(query, key, _rope_matrix(rope))
    if module.kv_cache is not None:
        if cache_position is None:
            raise ValueError("cache_position is required when KV cache is enabled")
        key, value = module.kv_cache.update(cache_position, key, value)

    preferred_attention = module.comfy_attention
    if module.comfy_decode_attention is not None:
        key = key[..., :attention_mask.shape[-1], :]
        value = value[..., :attention_mask.shape[-1], :]
        if batch == 1 and length == 1:
            # Single-item processor prompts have no padding; the mask ends at the current key.
            attention_mask = None
            preferred_attention = module.comfy_decode_attention
    if attention_mask is not None and attention_mask.dtype == torch.bool:
        attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype).masked_fill_(
            ~attention_mask, float("-inf"))
    gqa = {"enable_gqa": True} if module.n_head != module.n_local_heads else {}
    output = optimized_attention(
        AttentionTensorContainer(query),
        AttentionTensorContainer(key),
        AttentionTensorContainer(value),
        module.n_head,
        mask=attention_mask,
        skip_reshape=True,
        skip_output_reshape=True,
        preferred_attention=preferred_attention,
        **gqa,
    )
    return module.wo(output.transpose(1, 2).contiguous().view(batch, length, query_size))


def _native_linear(module):
    native = operations.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        device="meta",
        dtype=module.weight.dtype,
    )
    native.weight = module.weight
    if module.bias is not None:
        native.bias = module.bias
    return native.eval()


def _native_embedding(module):
    native = operations.Embedding(
        module.num_embeddings,
        module.embedding_dim,
        padding_idx=module.padding_idx,
        device="meta",
        dtype=module.weight.dtype,
    )
    native.weight = module.weight
    return native.eval()


def _native_rmsnorm(module):
    native = operations.RMSNorm(
        tuple(module.weight.shape),
        eps=module.eps,
        device="meta",
        dtype=module.weight.dtype,
    )
    native.weight = module.weight
    return native.eval()


def _patch_module(patcher, path, module):
    if isinstance(module, torch.nn.Linear):
        patcher.add_object_patch(path, _native_linear(module))
    elif isinstance(module, torch.nn.Embedding):
        patcher.add_object_patch(path, _native_embedding(module))
    elif not isinstance(module, torch.nn.Identity):
        patcher.add_object_patch(path, _native_rmsnorm(module))


def _patch_attention(patcher, path, module, backend):
    attention = ComfyAttention()
    decode_attention = None
    if backend != "auto":
        function = get_attention_function(backend, None)
        if function is None:
            raise RuntimeError(f"[Audio8] Attention backend is not registered: {backend}")
        decode_attention = ComfyAttention()
        decode_attention.function = function
        if backend == "comfy_kitchen_int8":
            attention.function = function

            def kitchen_decode(q, k, v, heads, **kwargs):
                # comfy-kitchen 0.2.37's DLPack contiguity check rejects Q=1.
                # Two identical, independent queries preserve the first result.
                if isinstance(q, AttentionTensorContainer):
                    q, k, v = q.take(), k.take(), v.take()
                result = function(torch.cat((q, q), dim=-2), k, v, heads, **kwargs)
                return result[..., :1, :]

            decode_attention.function = kitchen_decode
    patcher.add_object_patch(path + ".comfy_attention", attention)
    patcher.add_object_patch(path + ".comfy_decode_attention", decode_attention)

    def forward(x, rope, attention_mask, cache_position=None):
        return _native_attention(module, x, rope, attention_mask, cache_position)

    patcher.add_object_patch(path + ".forward", forward)
    _patch_module(patcher, path + ".wqkv", module.wqkv)
    _patch_module(patcher, path + ".wo", module.wo)
    if module.qk_norm:
        _patch_module(patcher, path + ".q_norm", module.q_norm)
        _patch_module(patcher, path + ".k_norm", module.k_norm)


def _patch_block(patcher, path, block, backend="auto"):
    _patch_attention(patcher, path + ".attention", block.attention, backend)
    _patch_module(patcher, path + ".ffn_norm", block.ffn_norm)
    _patch_module(patcher, path + ".attention_norm", block.attention_norm)
    _patch_module(patcher, path + ".feed_forward.w1", block.feed_forward.w1)
    _patch_module(patcher, path + ".feed_forward.w2", block.feed_forward.w2)
    _patch_module(patcher, path + ".feed_forward.w3", block.feed_forward.w3)


def _native_slow_step(model, input_ids, cache_position, position_ids, attention_mask, output_projection,
                      trim_cache=False):
    hidden = model._embed(input_ids)
    rope = model.freqs_cis[position_ids]
    key_length = attention_mask.shape[-1] if trim_cache else model.config.max_seq_len
    mask = model._causal_mask(attention_mask, cache_position, key_length)
    queue = comfy.model_prefetch.make_prefetch_queue(list(model.layers), hidden.device, {})
    try:
        for layer in model.layers:
            def core(layer=layer):
                nonlocal hidden
                hidden = layer(hidden, rope, mask, cache_position)

            comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core)
    finally:
        comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, None)
    hidden = hidden[:, -1:]
    normalized = model.norm(hidden)
    logits = output_projection(normalized)[:, -1]
    fast_hidden = normalized if model.config.norm_fastlayer_input else hidden
    return logits, fast_hidden


def _native_fast_step(model, hidden, position):
    cache_position = torch.tensor([position], device=hidden.device, dtype=torch.long)
    rope = model.fast_freqs_cis[cache_position]
    key_mask = torch.ones(
        (hidden.shape[0], model.config.num_codebooks),
        device=hidden.device,
        dtype=torch.bool,
    )
    mask = model._causal_mask(key_mask, cache_position, model.config.num_codebooks)
    queue = comfy.model_prefetch.make_prefetch_queue(list(model.fast_layers), hidden.device, {})
    try:
        for layer in model.fast_layers:
            def core(layer=layer):
                nonlocal hidden
                hidden = layer(hidden, rope, mask, cache_position)

            comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core)
    finally:
        comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, None)
    return model.fast_output(model.fast_norm(hidden))[:, -1]


@contextmanager
def comfy_audio8_ops(bundle, attention_backend="auto"):
    """Temporarily route Audio8 blocks through ComfyUI's shared operations."""

    model = bundle.model
    patcher = comfy.model_patcher.ModelPatcher(model, bundle.device, torch.device("cpu"))
    output_projection = operations.Linear(
        model.config.dim, model.config.vocab_size, bias=False,
        device="meta", dtype=model.embeddings.weight.dtype)
    output_projection.weight = model.embeddings.weight
    output_projection.eval()
    patcher.add_object_patch("_slow_step", lambda *args: _native_slow_step(
        model, *args, output_projection=output_projection, trim_cache=attention_backend != "auto"))
    patcher.add_object_patch("_fast_step", lambda hidden, position: _native_fast_step(model, hidden, position))

    for path, module in (
        ("embeddings", model.embeddings),
        ("codebook_embeddings", model.codebook_embeddings),
        ("fast_embeddings", model.fast_embeddings),
        ("fast_project_in", model.fast_project_in),
        ("fast_output", model.fast_output),
        ("norm", model.norm),
        ("fast_norm", model.fast_norm),
    ):
        _patch_module(patcher, path, module)
    for index, block in enumerate(model.layers):
        _patch_block(patcher, f"layers.{index}", block, attention_backend)
    for index, block in enumerate(model.fast_layers):
        _patch_block(patcher, f"fast_layers.{index}", block)

    try:
        patcher.patch_model(load_weights=False)
        yield
    finally:
        patcher.unpatch_model(unpatch_weights=False)
        patcher.object_patches.clear()
