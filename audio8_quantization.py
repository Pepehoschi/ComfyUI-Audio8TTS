"""Optional int8 weight quantisation for the Audio8 language model.

Default is unchanged (bfloat16). ``int8_convrot`` keeps bfloat16 activations and
rotates + groups each eligible ``nn.Linear`` weight through comfy-kitchen's
``TensorWiseINT8Layout`` so the stored weights stay int8. This layer only owns
the weight conversion; runtime, presets and the ComfyUI adapter are untouched.
"""

import logging

import torch

logger = logging.getLogger(__name__)

INT8_CONVROT = "int8_convrot"
INT8_CONVROT_GROUP_SIZE = 64


def quantize_model_int8_convrot(model, group_size=INT8_CONVROT_GROUP_SIZE):
    """Quantise every eligible ``nn.Linear`` of ``model`` in place.

    Eligible means ``in_features`` is divisible by ``group_size``; other
    modules keep their original float weights. The original weight tensor is
    discarded and replaced by the packed int8 layout, and the module forward
    dequantises on the fly, matching the measured standalone reference runs.
    Returns the number of quantised modules.
    """

    try:
        from comfy_kitchen.tensor import TensorWiseINT8Layout
    except ImportError as exc:
        raise RuntimeError(
            "[Audio8] int8_convrot requires the comfy-kitchen package from the "
            "ComfyUI environment (TensorWiseINT8Layout); no packages are installed."
        ) from exc

    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear) or module.weight is None:
            continue
        if module.in_features % group_size:
            continue
        packed, params = TensorWiseINT8Layout.quantize(
            module.weight.data, is_weight=True, per_channel=True, convrot=True,
            convrot_groupsize=group_size)
        bias = module.bias
        module.weight = None
        module.q_weight = packed
        module.q_params = params

        def int8_forward(x, packed=packed, params=params, bias=bias):
            return torch.nn.functional.linear(
                x, TensorWiseINT8Layout.dequantize(packed, params), bias)

        module.forward = int8_forward
        count += 1
        logger.debug("[Audio8] Quantised %s with group %d", name, group_size)
    if count:
        model.audio8_int8_quantised = True
        logger.info("[Audio8] INT8-ConvRot: quantised %d nn.Linear modules (group %d)",
                    count, group_size)
    elif not hasattr(model, "audio8_int8_quantised"):
        model.audio8_int8_quantised = False
    return count
