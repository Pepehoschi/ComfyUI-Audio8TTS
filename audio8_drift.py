"""Drift detection for private upstream steps this integration mirrors.

The CUDA graph adapters and the comfy_ops adapter mirror Audio8's private
modeling steps (_slow_step, _fast_step) and rely on _generate_codebooks
staying compatible. If a checkpoint updates those sources, the mirrored
behaviour can silently go wrong, so hash the model code at patch time and
warn when it no longer matches the reviewed sources.
"""

import ast
import hashlib
import inspect
import logging
import pathlib


logger = logging.getLogger(__name__)

MIRRORED_STEPS = ("_slow_step", "_fast_step", "_generate_codebooks")

# sha256 over each step's exact source segment, recorded from the shipped
# Edge0/Audio8-TTS-Realistic-0.6b checkpoint. Regenerate deliberately and
# re-verify the adapters before accepting a newer checkpoint.
REVIEWED_STEP_SHA256 = {
    "_slow_step": "8ed075edc8746edf865da2fbac1ce0dbd66549f05cc43cc942d8aa4670188ba6",
    "_fast_step": "4e62ad22e68231c99e695cda2bc03844df919795d20a73a7692a55fc6bf24995",
    "_generate_codebooks": "98f92ff79dced6ee574cc6d7be8709c220ba86ee3ff3c3aa9ff49d75ce00fd69",
}

_SOURCE_CACHE = {}
_WARNED = set()


def _model_source_tree(cls):
    key = (cls.__module__, cls.__qualname__)
    cached = _SOURCE_CACHE.get(key, False)
    if cached is not False:
        return cached
    parsed = None
    try:
        path = inspect.getsourcefile(cls)
        if path:
            source = pathlib.Path(path).read_text(encoding="utf-8")
            parsed = ast.parse(source), source
    except (OSError, SyntaxError, TypeError):
        parsed = None
    _SOURCE_CACHE[key] = parsed
    return parsed


def _step_segment(tree, source, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    return None


def check_step_sources(model):
    """Warn when a mirrored private step differs from the reviewed source."""

    parsed = _model_source_tree(type(model))
    if parsed is None:
        logger.warning("[Audio8] Cannot read the model source; drift of the mirrored "
                       "private steps (%s) against the reviewed integration is unchecked.",
                       ", ".join(MIRRORED_STEPS))
        return
    tree, source = parsed
    class_key = (type(model).__module__, type(model).__qualname__)
    for name in MIRRORED_STEPS:
        segment = _step_segment(tree, source, name)
        if segment is None:
            key = (class_key, name, "missing")
            if key not in _WARNED:  # one warning per process, not one per generate call
                logger.warning("[Audio8] Model class has no %s in its source; the adapters "
                               "mirroring it were reviewed against a different checkpoint.", name)
                _WARNED.add(key)
            continue
        digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
        if digest == REVIEWED_STEP_SHA256[name]:
            continue
        key = (class_key, name, digest)
        if key in _WARNED:
            continue
        logger.warning(
            "[Audio8] Upstream %s differs from the reviewed checkpoint source "
            "(reviewed %s..., found %s...). The CUDA graph and comfy_ops adapters "
            "mirror this step and may no longer reproduce it; fall back to "
            "cuda_graphs=False/comfy_ops=False if audio behaviour changes.",
            name, REVIEWED_STEP_SHA256[name][:8], digest[:8])
        _WARNED.add(key)
