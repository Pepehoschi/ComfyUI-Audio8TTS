"""Optional int8 weight quantisation for the Audio8 language model.

Default is unchanged (bfloat16). ``int8_convrot`` keeps bfloat16 activations and
rotates + groups each eligible ``nn.Linear`` weight through comfy-kitchen's
``TensorWiseINT8Layout`` so the stored weights stay int8. This layer owns the
weight conversion in both directions: on-the-fly quantisation of a loaded model
and re-installation of pre-quantised pack tensors. Runtime, presets and the
ComfyUI adapter are untouched.
"""

import dataclasses
import logging

import torch

logger = logging.getLogger(__name__)

INT8_CONVROT = "int8_convrot"
INT8_CONVROT_GROUP_SIZE = 64


def _require_layout():
    try:
        from comfy_kitchen.tensor import TensorWiseINT8Layout
    except ImportError as exc:
        raise RuntimeError(
            "[Audio8] int8_convrot requires the comfy-kitchen package from the "
            "ComfyUI environment (TensorWiseINT8Layout); no packages are installed."
        ) from exc
    return TensorWiseINT8Layout


def _dtype_from_name(name):
    """Resolve the serialized str(torch.bfloat16) back to a torch dtype."""

    dtype = getattr(torch, str(name).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"[Audio8] Unknown int8 orig_dtype {name!r} in pack metadata")
    return dtype


def _install_int8_forward(module, packed, params, bias):
    """Turn one nn.Linear into its dequantising-on-the-fly form."""

    TensorWiseINT8Layout = _require_layout()
    module.weight = None
    module.q_weight = packed
    module.q_params = params

    def int8_forward(x, packed=packed, params=params, bias=bias):
        return torch.nn.functional.linear(
            x, TensorWiseINT8Layout.dequantize(packed, params), bias)

    module.forward = int8_forward


def quantize_model_int8_convrot(model, group_size=INT8_CONVROT_GROUP_SIZE):
    """Quantise every eligible ``nn.Linear`` of ``model`` in place.

    Eligible means ``in_features`` is divisible by ``group_size``; other
    modules keep their original float weights. The original weight tensor is
    discarded and replaced by the packed int8 layout, and the module forward
    dequantises on the fly, matching the measured standalone reference runs.
    Returns the number of quantised modules.
    """

    TensorWiseINT8Layout = _require_layout()

    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear) or module.weight is None:
            continue
        if module.in_features % group_size:
            continue
        packed, params = TensorWiseINT8Layout.quantize(
            module.weight.data, is_weight=True, per_channel=True, convrot=True,
            convrot_groupsize=group_size)
        _install_int8_forward(module, packed, params, module.bias)
        count += 1
        logger.debug("[Audio8] Quantised %s with group %d", name, group_size)
    if count:
        model.audio8_int8_quantised = True
        logger.info("[Audio8] INT8-ConvRot: quantised %d nn.Linear modules (group %d)",
                    count, group_size)
    elif not hasattr(model, "audio8_int8_quantised"):
        model.audio8_int8_quantised = False
    return count


def collect_int8_convrot_state(model):
    """Serialise the installed int8_convrot layouts of a quantised model.

    Returns (tensors, meta): tensors maps module path -> comfy-kitchen
    state_dict_tensors suffix mapping ("" packed int8 plus "_scale"), all on
    CPU; meta maps module path -> the scalar Params fields needed to rebuild
    the Params dataclass verbatim.
    """

    TensorWiseINT8Layout = _require_layout()
    tensors = {}
    meta = {}
    for name, module in model.named_modules():
        packed = getattr(module, "q_weight", None)
        if packed is None:
            continue
        parts = TensorWiseINT8Layout.state_dict_tensors(packed, module.q_params)
        tensors[name] = {suffix: tensor.detach().to("cpu").contiguous()
                         for suffix, tensor in parts.items()}
        params = module.q_params
        entry: dict = {"orig_dtype": str(params.orig_dtype)}
        for field in dataclasses.fields(params):
            if field.name in ("scale", "orig_dtype"):
                continue
            value = getattr(params, field.name)
            entry[field.name] = list(value) if isinstance(value, tuple) else value
        meta[name] = entry
    return tensors, meta


def apply_int8_convrot_state(model, tensors, meta, device, dtype=None):
    """Re-install pre-quantised pack tensors onto freshly built modules.

    ``tensors`` maps module path -> {"": packed int8, "_scale": scale} (the
    suffix mapping produced by collect_int8_convrot_state); ``meta`` carries
    the scalar Params fields from the same call. The float weight parameter of
    each listed module must already be removed from the module and from the
    state dict used for load_state_dict. ``dtype``, when given, overrides the
    stored orig_dtype so dequantised weights match the runtime compute dtype.
    Returns the number of installed modules.
    """

    TensorWiseINT8Layout = _require_layout()
    param_fields = {field.name for field in dataclasses.fields(TensorWiseINT8Layout.Params)}
    count = 0
    for name, entry in tensors.items():
        fields = meta[name]
        if tuple(fields["orig_shape"]) != tuple(entry[""].shape):
            raise ValueError(
                f"[Audio8] Pack int8 shape mismatch for {name}: "
                f"{entry[''].shape} vs {fields['orig_shape']}")
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"[Audio8] Pack int8 target {name} is no nn.Linear")
        kwargs = {key: value for key, value in fields.items()
                  if key in param_fields and key not in ("scale", "orig_dtype")}
        params = TensorWiseINT8Layout.Params(
            scale=entry["_scale"].to(device=device, dtype=torch.float32),
            orig_dtype=dtype if dtype is not None else _dtype_from_name(fields["orig_dtype"]),
            **kwargs)
        _install_int8_forward(module, entry[""].to(device=device), params, module.bias)
        count += 1
    model.audio8_int8_quantised = bool(count)
    return count
