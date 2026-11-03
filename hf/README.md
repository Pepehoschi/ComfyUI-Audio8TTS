---
license: apache-2.0
base_model: Edge0/Audio8-TTS-Realistic-0.6b
pipeline_tag: text-to-speech
tags:
- text-to-speech
- audio
- voice-cloning
- safetensors
- comfyui
- int8
- quantization
---

# Audio8-TTS-Realistic-0.6b — ComfyUI model packs (BF16 + INT8)

Single-file **model packs** of [Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b)
for use inside ComfyUI. Each pack folds the original model directory (language
model, 44.1 kHz codec, tokenizer and processor files) into one `.safetensors`
file that ComfyUI's `diffusion_models` picker can list and load.

This is a community re-packaging, **not an official Edge0 release**. The
weights are unchanged from upstream apart from the INT8 weight quantisation
described below.

ComfyUI nodes: [Pepehoschi/ComfyUI-Audio8TTS](https://github.com/Pepehoschi/ComfyUI-Audio8TTS)
(public GitHub repository, includes example workflows and setup instructions).

## Files

| File | Size | Contents |
| --- | --- | --- |
| `Audio8-TTS-Realistic-0.6b-BF16.safetensors` | 2,567,214,504 bytes | Original BF16 weights, packed lossless. SHA256 `d9720afccadca7f162bc780775b440d5f1bbc98bf1175e62281e31c75d410e42` |
| `Audio8-TTS-Realistic-0.6b-int8convrot.safetensors` | 2,147,558,576 bytes | ConvRot INT8 weight quantisation (8-bit weights + per-channel scales, BF16 activations), 16% smaller. Metadata `audio8_quant=int8_convrot`. SHA256 `cc13f4b0b6e1cd2712bb1339819988574e7e9c3ab1fa8bdc21cbddc7202e3306` |

Packing is lossless and deterministic: repacking the same source directory
reproduces identical tensor bytes.

## Usage in ComfyUI

1. Install the [ComfyUI-Audio8TTS](https://github.com/Pepehoschi/ComfyUI-Audio8TTS)
   custom nodes (git clone into `ComfyUI/custom_nodes/`).
2. Put one of the packs into `ComfyUI/models/diffusion_models/audio8/`.
3. Add an **Audio8 Model Loader** node and pick the file from its dropdown.
   The loader selects the compute path from the pack metadata
   (`audio8_quant=int8_convrot`), not from the file name; `dtype="auto"` is
   the recommended setting.
4. Build a graph with the other Audio8 nodes (`Audio8 TTS`,
   `Audio8 Save Voice`, `Audio8 Load Voice`, `Audio8 Encode Reference`, …).
   Two ready-made templates ship under the ComfyUI template browser
   (category *ComfyUI-Audio8TTS*).

The original unpacked Edge0 directory keeps working as an inference source;
these packs only provide the native single-file ComfyUI loading path.

## Measured behaviour

Local measurements on a ComfyUI cu128 instance (256 frames, greedy sampling,
`cuda_graphs=true`, `cuda_graph_scope=frame`, `comfy_ops=false`):

- BF16 pack: ~54 tok/s.
- INT8 pack: 30.4 tok/s (eager 32.62 s, `fast` 16.44 s, `frame` 8.43 s for
  256 frames). BF16 remains the faster path; INT8 mainly saves memory and
  download size.
- INT8 sampling is bit-identical across eager / `fast` / `frame` modes for
  the same pack, and the INT8 pack is bit-identical to on-the-fly INT8
  quantisation of the BF16 pack (maxdiff 0.0 over a 491,520-sample render).
- The INT8 quantisation itself is deterministic across fresh loads.

## Provenance and licence

- Base model and architecture: [Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b),
  Apache-2.0. The upstream `LICENSE` and `NOTICE` files are included in this
  repository unchanged.
- ComfyUI node code, packing tooling and tests:
  [Pepehoschi/ComfyUI-Audio8TTS](https://github.com/Pepehoschi/ComfyUI-Audio8TTS).
- These packs are therefore distributed under Apache-2.0 as well.

## Limitations

- The upstream model supports 23 languages; per-language quality varies —
  see the upstream model card for evaluations.
- INT8 is weight-only quantisation; activations stay BF16. No accuracy
  claims beyond the measurements above are made.
- The packs target the ComfyUI node workflow; plain Transformers inference
  should use the original upstream repository.
