# ComfyUI Audio8 TTS

Work in progress: native nodes for `Edge0/Audio8-TTS-Realistic-0.6b`.

## Environment

The project targets the existing ComfyUI cu128 installation. Do not separately install or replace Torch/Torchaudio.

`requirements.txt` deliberately lists only `transformers>=4.57.0,<6` and `safetensors>=0.4`. Torch, Torchaudio and NumPy stay out on purpose: ComfyUI owns the PyTorch/CUDA build of its installation (cu128 here), and a pip-resolved `torch` entry would replace that build with a generic CUDA wheel and break the install. This boundary also applies to any future `pyproject.toml` for registry publishing - keep dependencies dynamic from `requirements.txt` and never add a torch-family entry. A test in `tests/test_comfy_contract.py` fails if a dependency file ever gains one.

The integration supports Transformers `>=4.57.0,<6`. The complete local model, processor and codec have been tested with Transformers 5.9.0 and Torch 2.8.0+cu128 on an RTX 4090. No packages were changed. Model files were copied from the existing HuggingFace cache.

Load an Audio8 **pack** - one `.safetensors` file - through the
`Audio8 Model Loader` node. Its required `ckpt_name` input is the same
picker the native `Load Diffusion Model` node uses: it lists the files in
the `diffusion_models` folders (`ComfyUI/models/diffusion_models` plus
every root mapped in `extra_model_paths.yaml`, e.g. a Stability-Matrix
shared-models folder), shows them as a searchable folder tree and
serializes only the relative path into the workflow.

Create a pack once from a complete model directory (the HuggingFace
download form: `config.json`, `codec.pth`, weights, tokenizer and custom
Python files) with the `Audio8 Pack Model` node. It writes the pack to
the `audio8/` subfolder of the `diffusion_models` root that already holds
one, e.g.:

```text
ComfyUI/models/diffusion_models/audio8/Audio8-TTS-Realistic-0.6b.safetensors
```

The source directory stays untouched and keeps serving as the converter
input; after packing it has no other function inside the nodes and can
remain on disk as a backup of the originals. Model-directory loading was
removed from the loader: a pack carries every weight, config and
tokenizer in one file, and loading instantiates the node's audited
Apache-2.0 architecture copy (`audio8_arch/`, sha256 recorded in
`audio8_arch/__init__.py`, attribution in `NOTICE.upstream`) directly and
applies weights via `load_state_dict` - no `trust_remote_code`, no
dynamic modules, no Auto* registration anywhere. Do not register the
`audio8_arch` classes with the transformers Auto* factories:
transformers 5.x then silently keeps constructor-initialized weights
instead of loading the checkpoint (verified on 5.9.0).

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

Load `examples/Audio8-Start.json` for a frontend canvas workflow (Model Loader, Load Voice, TTS, Save Audio). On this machine it selects `Audio8-Referenz.safetensors`, encoded from the supplied reference recording. Other installations must create their own voice preset first.

- Audio8 Model Loader: shared model and processor; `ckpt_name` picker over the `diffusion_models` folders; CPU uses FP32.
- Audio8 Pack Model: folds a model directory into one `.safetensors` pack under `audio8/` in the `diffusion_models` folder that already holds one; `quantize="int8_convrot"` writes a separate `…-int8convrot.safetensors`.
- Audio8 Encode Reference: ComfyUI audio plus its transcript becomes reusable codec codes.
- Audio8 Save Voice: atomically saves a reference under `models/audio8_voices/`.
- Audio8 Load Voice: loads codes and transcript without the original audio file.
- Audio8 TTS: produces native ComfyUI AUDIO, with an optional reference, automatic long-text chunking and per-chunk NO_EOS retry.

Connect the model loader to Encode Reference and TTS. Connect Load Audio to Encode Reference, then its reference output to TTS or Save Voice. For saved voices, connect Load Voice to TTS. Connect TTS to ComfyUI Save Audio.

## Current verification

Data validation, preset roundtrips and mocked loader/runtime tests run without weights. The first inference checks missed corrupted non-persistent RoPE buffers after Transformers 5 loading; nonempty audio alone was insufficient. The loader now rebuilds these buffers using Audio8's own helper. Codec encoding and decoding also disable TorchScript optimization locally because PyTorch 2.8.0+cu128 changed BF16 results after warmup. No global Torch configuration is changed. The pack format was verified end to end on a live server: the real model packed through the node, then two anchor generations through the pack were sample-identical to the directory baseline (524288 samples, maxdiff 0.0).

The corrected CUDA/BF16 path passes repeated-encoding, preset-roundtrip and Direct-vs-Cached tests with the supplied recording, reaches EOS, and produces an audible-level waveform. Listening evaluation remains necessary. Restart ComfyUI after updating: its cached model and imported runtime otherwise retain the old behavior. The invalid generated test preset has been moved out of the voices directory.

Long-text defaults are 130 target / 150 maximum characters, 180 ms between chunks and 350 ms at paragraph boundaries. One generator advances through the whole run, including retries. Reference codes are reused across chunks. Short text keeps the direct path; `chunking=disabled` sends the whole text to Audio8. Each decoded chunk receives a 5 ms fade-out before its pause; optional `fade_out_ms=0` disables it. Audio length is preserved. No trailing pause or automatic crossfade is added.

The remaining example files in `examples/` are API prompt graphs, not frontend canvas workflows; `Audio8-Start.json` above is the only canvas file. `text_to_speech.api.json` synthesizes text, `save_voice.api.json` encodes and saves a reference, and `saved_voice_tts.api.json` loads a saved voice. Replace the example audio filename and transcript with matching local data. A saved-voice graph has completed on the local ComfyUI server and saved FLAC through Save Audio. A second request reused the cached loader and voice nodes. An actual server restart remains unverified.

## Performance

The optional `comfy_ops` TTS input routes Audio8's inference layers through shared ComfyUI Linear, Embedding and RMSNorm operations, shared RoPE, and ComfyUI's selected attention implementation. The adapter preserves causal masking and grouped-query heads, shares checkpoint parameters, and uses a temporary ModelPatcher. The original processor, codec and generation rules remain in use. This does not convert or quantize the checkpoint and does not add full ComfyUI model offloading.

It defaults to off. Enable it after restarting ComfyUI; it can be combined with `cuda_graphs`. Numerical results can differ from the original path because normalization and attention kernels round differently. The tested BF16 prefill logits had 0.92% relative RMS difference; complete sample generation reached EOS on both paths but produced different frame counts.

With `comfy_ops=True`, `attention_backend` selects `auto`, `flash`, or `comfy_kitchen_int8`. `auto` preserves ComfyUI's standard attention selection. The explicit choices optimize the slow decoder for this node's single-item processor prompts: cache keys end at the valid logical length, so a one-query decode needs no causal mask. Prefill and the fast decoder keep their masks. Flash is selected for slow decode; Kitchen INT8 is selected for prefill and slow decode. Kitchen's current single-query layout issue is handled by duplicating the independent query and taking its first result. Both choices require the respective installed ComfyUI backend.

The installed cu128 setup executes real Flash and Kitchen INT8 CUDA attention kernels, verified by profiling the complete model. ComfyUI's warning about the Kitchen CUDA operation registry does not describe the availability of this separate attention entry point. Earlier blanket statements that no Kitchen CUDA acceleration was possible on cu128 were incorrect.

Five interleaved runs on the RTX 4090, 32 fixed frames and graph setup included:

| Path | Median generation time |
| --- | --- |
| Original model + CUDA Graphs | 1.172 s |
| Comfy operations, auto + CUDA Graphs | 1.162 s |
| Comfy operations, Flash + CUDA Graphs | 1.056 s |
| Comfy operations, Kitchen INT8 + CUDA Graphs | 1.098 s |

For speed without changing the sound, keep `comfy_ops=False` and use `cuda_graphs=True` with `cuda_graph_scope="frame"`: measured about four times faster and bit-identical audio. The attention backends add little on top and change the result: `flash` and `comfy_kitchen_int8` alter the sampled trajectory, and Kitchen INT8 quantises query, key and value. `cuda_graph_scope="frame"` is deliberately not combined with those two backends, the node logs a warning and falls back to `"fast"`. Model/codec loading and audio saving are outside these generation timings.

`benchmarks/native_ops.py --output <directory>` compares prefill results and four variants (original/native, with/without CUDA Graphs), writes timings to JSON and produces comparison WAV files. It loads only local model and voice files.

`benchmarks/compare_attention.py --profile --output <directory>` performs the interleaved attention comparison and records CUDA kernel evidence. `benchmarks/attention_backends.py` measures the registered implementations at representative Audio8 attention shapes.

`benchmarks/audio_regression.py --record` followed by `--check` is the listening test you can run unattended. It sends one fixed text through four attention levels and three graph scopes to the running server and fails on two things: output whose peak, RMS or high frequency energy leaves the clean band, and any backend whose audio changes between graph scopes. A recorded reference additionally pins the waveform hash for strict same-machine comparison. It exists because one configuration degraded into audible noise while still reporting success and the expected frame count — only the waveform revealed it.

After restarting ComfyUI, enable the optional `cuda_graphs` input on Audio8 TTS to replay its decoder with CUDA Graphs. It defaults to off and is ignored on CPU. `cuda_graph_scope` selects the extent: `fast` (default) replays only the fast decoder, `frame` replays the whole frame, so one frame becomes two graph replays. The patch uses ComfyUI's ModelPatcher, preserves sampling, and releases graph tensors after each generation attempt; retries and chunks receive fresh graphs. It requires no Torch update or additional dependencies. Capture has an upfront cost, so very short outputs can benefit less.

On the local RTX 4090 / Torch 2.8.0+cu128 / BF16 setup, a 75-frame utterance took a median 5.00 seconds with the standard path and 2.95 seconds with CUDA Graphs (two runs each, model already loaded, graph capture included). Generated codes were exactly equal. These are generation timings; initial model/codec loading and audio saving are separate.

Measured again on the same setup with 96 tokens and greedy decoding, so all three variants are directly comparable: 17.21 s without graphs, 8.28 s with `cuda_graph_scope=fast`, 2.82 s with `frame`. Generated codes stayed exactly equal to the ungraphed path. With the node default sampling the same comparison was 8.64 s without graphs against 1.99 s with `frame`. Sampling draws its random numbers inside the graph, so one seed yields other codes than the ungraphed path while a different seed still changes the output. `frame` reserves more graph memory per generation attempt.

Verified end to end through the running server, saved audio measured instead of codes (256 token budget, same text and seed, model and voice cached): 13.7 tokens/s without graphs, 53.9 with `cuda_graph_scope="frame"`, 28.0 with `comfy_ops=True attention_backend=flash` and `"fast"`. The three configurations without a maskless attention backend produced **bit-identical waveforms**, repeats included. High frequency energy of every clean sample sat between 0.22 and 0.28, so a listener comparison is meaningful; `frame` reserves more graph memory per generation attempt.

`benchmarks/profile_generation.py` records a reproducible local benchmark and optional CPU/CUDA operator profile. It defaults to the local `Audio8-Referenz.safetensors` voice and never downloads weights:

```powershell
& 'D:\AI\ComfyUI_windows_portable\python_embeded\python.exe' benchmarks/profile_generation.py --tokens 256 --profile-tokens 0 --skip-kernel-experiments --output 'D:\AI\ComfyUI_windows_portable\ComfyUI\output\audio8_performance'
```

## Tests

From this project directory, run with ComfyUI's Python:

```powershell
& 'D:\AI\ComfyUI_windows_portable\python_embeded\python.exe' -m unittest discover -s tests
```

The real model test is skipped by default. Once compatible Transformers, complete local weights and a reference recording are available, opt in with:

```powershell
$env:AUDIO8_RUN_INTEGRATION = '1'
$env:AUDIO8_TEST_REFERENCE = 'D:\path\reference.wav'
$env:AUDIO8_TEST_TRANSCRIPT = 'Exact words spoken in the reference recording.'
& 'D:\AI\ComfyUI_windows_portable\python_embeded\python.exe' -m unittest discover -s tests -p test_model_integration.py
```

That test uses the existing SoundFile installation to read the reference, checks the preset roundtrip and compares generated codes for direct versus cached reference conditioning with greedy decoding. It performs no model downloads.

See [PROGRESS.md](PROGRESS.md) for the current state and [ImplementationPlan.md](ImplementationPlan.md) for remaining work.

## Limitations

- ComfyUI cancellation takes effect between text chunks; a generation that already
  started runs to its end.
- Model, processor and codec stay in the node cache, not in ComfyUI's memory manager:
  unload and empty-cache operations do not release them, and their VRAM is invisible
  to ComfyUI's memory accounting until the process exits.
- The model loader reloads when the checkpoint directory changes (file count, total
  size or newest modification time); a swapped checkpoint is picked up on the next run.
