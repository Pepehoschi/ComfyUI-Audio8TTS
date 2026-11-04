# ComfyUI Audio8 TTS

Native ComfyUI nodes for `Edge0/Audio8-TTS-Realistic-0.6b`.
This repository is an independent ComfyUI integration, not a fork. Model and
reference implementation are by Edge0: [Edge0-AI/Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS),
base model: [Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b).

## Installation

Install **Audio8 TTS** (`comfyui-audio8tts`) from the ComfyUI Registry through ComfyUI Manager, then restart ComfyUI. Install a model pack separately by following [Models](#models).

For a manual install, clone the repository into `ComfyUI/custom_nodes` and install `requirements.txt` using the Python environment that runs ComfyUI. Audio8 requires Transformers `>=4.57.0,<6` and Safetensors `>=0.4`. It uses the PyTorch build provided by your ComfyUI installation.

## Models

The nodes load packs from
[Dhrog/Audio8-TTS-Realistic-0.6b-ComfyUI-Packs](https://huggingface.co/Dhrog/Audio8-TTS-Realistic-0.6b-ComfyUI-Packs)
(model card, SHA256 hashes, measurements). Download **one** file directly:

| Pack | Direct download | Use when |
| --- | --- | --- |
| BF16 | [Audio8-TTS-Realistic-0.6b-BF16.safetensors](https://huggingface.co/Dhrog/Audio8-TTS-Realistic-0.6b-ComfyUI-Packs/resolve/main/Audio8-TTS-Realistic-0.6b-BF16.safetensors) | Recommended starting point. See [performance](#performance). |
| INT8 | [Audio8-TTS-Realistic-0.6b-int8convrot.safetensors](https://huggingface.co/Dhrog/Audio8-TTS-Realistic-0.6b-ComfyUI-Packs/resolve/main/Audio8-TTS-Realistic-0.6b-int8convrot.safetensors) | About 16% smaller download than BF16. Uses INT8 ConvRot weights. |

Place the downloaded file in an `audio8/` subfolder of any
`diffusion_models` root, exactly:

```text
ComfyUI/models/diffusion_models/audio8/Audio8-TTS-Realistic-0.6b-BF16.safetensors
```

(or the same `audio8/` subfolder under a root mapped in
`extra_model_paths.yaml`). Press `R` in ComfyUI to refresh the file
catalogue, then pick the file in `Audio8 Model Loader` with
`dtype="auto"`. Nothing else is required: a pack is self-contained
(language-model weights, codec, config, tokenizer and custom architecture
metadata in one file). Quantisation is chosen from the pack's
`audio8_quant` metadata key, never from the file name, so renaming the
pack only changes how it is labelled.

The packs are derived from
[`Edge0/Audio8-TTS-Realistic-0.6b`](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b)
(Apache-2.0; `LICENSE` and `NOTICE` ship in the HF repo; model project page:
[Edge0-AI/Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS)). A community
re-packaging, not an official Edge0 release.

Create a pack once from a complete model directory (the HuggingFace
download form: `config.json`, `codec.pth`, weights, tokenizer and custom
Python files) with the `Audio8 Pack Model` node as an alternative to
downloading. It writes the pack to the `audio8/` subfolder of the
`diffusion_models` root that already holds one, e.g.:

```text
ComfyUI/models/diffusion_models/audio8/Audio8-TTS-Realistic-0.6b-BF16.safetensors
```

The loader uses the bundled Apache-2.0 architecture in `audio8_arch/` and reads weights directly from the pack. See `NOTICE.upstream` for attribution.

### Pack format

A pack folds one model directory into one `.safetensors` file, the way
ComfyUI's own checkpoint loaders work: language-model tensors under
`model.`, codec tensors under `codec.`, and the config/tokenizer JSONs
as safetensors header metadata. Loading validates the `audio8_format`
metadata key, so picking a foreign `.safetensors` (for example a real
UNet from the same folder) fails early with a clear error. Verified
sample-identical to the former directory path (maxdiff 0.0 against the
BF16 anchor, through the native picker). The metadata key `audio8_quant`
marks quantised packs: with `quantize="int8_convrot"` the Pack Model
node writes a separate `…-int8convrot.safetensors` next to the BF16 pack,
keeping each quantised `nn.Linear` as an int8 tensor plus a float32
`.weight_scale` companion. The loader reinstalls those layouts directly
(`dtype=auto` or `int8_convrot` both work), so nothing is requantised at
load time and runs from such a pack are reproducible; measured
sample-identical to on-the-fly `dtype=int8_convrot` (maxdiff 0.0).

## Nodes

Load `example_workflows/Audio8_Start.json` for a frontend canvas workflow (Model Loader, Load Voice, TTS, Save Audio). Select your model pack and voice preset. To create a preset, connect Load Audio to Audio8 Encode Reference, enter the transcript, and connect the reference output to Audio8 Save Voice. `example_workflows/Audio8-Voice-Tags.json` shows per-segment emotional voices with a voice set and a tagged script. Both files appear in ComfyUI's Workflow Templates browser under `ComfyUI-Audio8TTS`; the same-name `.jpg` next to each JSON is the card preview (the frontend always requests `.jpg`).

- Audio8 Model Loader: shared model and processor; `ckpt_name` picker over the `diffusion_models` folders; CPU uses FP32.
- Audio8 Pack Model: folds a model directory into one `.safetensors` pack under `audio8/` in the `diffusion_models` folder that already holds one; `quantize="int8_convrot"` writes a separate `…-int8convrot.safetensors`.
- Audio8 Encode Reference: ComfyUI audio plus its transcript becomes reusable codec codes.
- Audio8 Save Voice: atomically saves a reference under `models/audio8_voices/`.
- Audio8 Load Voice: loads codes and transcript without the original audio file.
- Audio8 Voice Set / Audio8 Voice Set Add: build a free `{tag → voice reference}` set in the graph; a later duplicate tag automatically gains a `_1`, `_2`, … suffix instead of failing.
- Audio8 Script: multiline text plus the wired voice set; `[tag]` switches the reference until the next tag, `[[`/`]]` escape literal brackets, unknown tags stay verbatim with a warning. Outputs the parsed script and a `log` string.
- Audio8 TTS: produces native ComfyUI AUDIO, with an optional reference, automatic long-text chunking and per-chunk NO_EOS retry; accepts the parsed `script` instead of plain `text` for per-segment voices.

Connect the model loader to Encode Reference and TTS. Connect Load Audio to Encode Reference, then its reference output to TTS or Save Voice. For saved voices, connect Load Voice to TTS. Connect TTS to ComfyUI Save Audio.

Long-text defaults are 130 target / 150 maximum characters, 180 ms between chunks and 350 ms at paragraph boundaries. One generator advances through the whole run, including retries. Reference codes are reused across chunks. Short text keeps the direct path; `chunking=disabled` sends the whole text to Audio8. Each decoded chunk receives a 110 ms fade-out before its pause; `fade_out_ms=0` disables it. Audio length is preserved. No trailing pause or automatic crossfade is added.

## Performance

The optional `comfy_ops` TTS input routes Audio8's inference layers through shared ComfyUI Linear, Embedding and RMSNorm operations, shared RoPE, and ComfyUI's selected attention implementation. The adapter preserves causal masking and grouped-query heads, shares checkpoint parameters, and uses a temporary ModelPatcher. The original processor, codec and generation rules remain in use. This does not convert or quantize the checkpoint and does not add full ComfyUI model offloading.

It defaults to off. Enable it after restarting ComfyUI; it can be combined with `cuda_graphs`. Numerical results can differ from the original path because normalization and attention kernels round differently. The tested BF16 prefill logits had 0.92% relative RMS difference; complete sample generation reached EOS on both paths but produced different frame counts.

With `comfy_ops=True`, `attention_backend` selects `auto`, `flash`, or `comfy_kitchen_int8`. `auto` preserves ComfyUI's standard attention selection. The explicit choices optimize the slow decoder for this node's single-item processor prompts: cache keys end at the valid logical length, so a one-query decode needs no causal mask. Prefill and the fast decoder keep their masks. Flash is selected for slow decode; Kitchen INT8 is selected for prefill and slow decode. Kitchen's current single-query layout issue is handled by duplicating the independent query and taking its first result. Both choices require the respective installed ComfyUI backend.

The `cuda_graphs` input on Audio8 TTS is on by default and replays its decoder with CUDA Graphs; it is ignored on CPU. `cuda_graph_scope` selects the extent: `frame` (default) replays the whole frame, so one frame becomes two graph replays, `fast` replays only the fast decoder. The patch uses ComfyUI's ModelPatcher, preserves sampling, and releases graph tensors after each generation attempt; retries and chunks receive fresh graphs. It requires no Torch update or additional dependencies. Capture has an upfront cost, so very short outputs can benefit less.

For the fastest measured configuration, enable CUDA Graphs, select `cuda_graph_scope="frame"`, and keep `comfy_ops=False`. Flash and Kitchen INT8 require compatible kernels. Selecting either with `frame` switches the graph scope to `fast` and logs a warning.

The benchmark used a BF16 pack on an RTX 4090 with PyTorch 2.8.0+cu128, a 256-token budget, and a cached model and voice. Results vary by hardware, backend and text.

| Configuration | Generation rate |
| --- | ---: |
| CUDA Graphs disabled | 13.7 tokens/s |
| Frame CUDA Graphs | 53.9 tokens/s |
| Flash attention with fast CUDA Graphs | 28.0 tokens/s |

Backend and sampling choices can change the generated audio. Frame graphs also use more VRAM during generation.

## Limitations

- ComfyUI cancellation takes effect between text chunks; a generation that already
  started runs to its end.
- Model, processor and codec stay in the node cache, not in ComfyUI's memory manager:
  unload and empty-cache operations do not release them, and their VRAM is invisible
  to ComfyUI's memory accounting until the process exits.
- The model loader checks the selected pack file size and modification time to detect changes.
