# ComfyUI Audio8 TTS

Work in progress: native nodes for `Edge0/Audio8-TTS-Realistic-0.6b`.

## Environment

The project targets the existing ComfyUI cu128 installation. Do not separately install or replace Torch/Torchaudio.

The integration supports Transformers `>=4.57.0,<6`. The complete local model, processor and codec have been tested with Transformers 5.9.0 and Torch 2.8.0+cu128 on an RTX 4090. No packages were changed. Model files were copied from the existing HuggingFace cache.

Place the complete model, including its custom Python files, tokenizer, weights and `codec.pth`, in:

```text
ComfyUI/models/audio8_tts/Audio8-TTS-Realistic-0.6b/
```

The loader uses local files only. Audio8's supplied Python code is loaded with `trust_remote_code=True`; use model files from a trusted source.

## Nodes

Load `examples/Audio8-Start.json` for a frontend canvas workflow (Model Loader, Load Voice, TTS, Save Audio). On this machine it selects `Audio8-Referenz.safetensors`, encoded from the supplied reference recording. Other installations must create their own voice preset first.

- Audio8 Model Loader: shared model and processor; CPU uses FP32.
- Audio8 Encode Reference: ComfyUI audio plus its transcript becomes reusable codec codes.
- Audio8 Save Voice: atomically saves a reference under `models/audio8_voices/`.
- Audio8 Load Voice: loads codes and transcript without the original audio file.
- Audio8 TTS: produces native ComfyUI AUDIO, with an optional reference, automatic long-text chunking and per-chunk NO_EOS retry.

Connect the model loader to Encode Reference and TTS. Connect Load Audio to Encode Reference, then its reference output to TTS or Save Voice. For saved voices, connect Load Voice to TTS. Connect TTS to ComfyUI Save Audio.

## Current verification

Data validation, preset roundtrips and mocked loader/runtime tests run without weights. The first inference checks missed corrupted non-persistent RoPE buffers after Transformers 5 loading; nonempty audio alone was insufficient. The loader now rebuilds these buffers using Audio8's own helper. Codec encoding and decoding also disable TorchScript optimization locally because PyTorch 2.8.0+cu128 changed BF16 results after warmup. No global Torch configuration is changed.

The corrected CUDA/BF16 path passes repeated-encoding, preset-roundtrip and Direct-vs-Cached tests with the supplied recording, reaches EOS, and produces an audible-level waveform. Listening evaluation remains necessary. Restart ComfyUI after updating: its cached model and imported runtime otherwise retain the old behavior. The invalid generated test preset has been moved out of the voices directory.

Long-text defaults are 130 target / 150 maximum characters, 180 ms between chunks and 350 ms at paragraph boundaries. One generator advances through the whole run, including retries. Reference codes are reused across chunks. Short text keeps the direct path; `chunking=disabled` sends the whole text to Audio8. Each decoded chunk receives a 5 ms fade-out before its pause; optional `fade_out_ms=0` disables it. Audio length is preserved. No trailing pause or automatic crossfade is added.

The example files in `examples/` are API prompt graphs, not frontend canvas workflows. `text_to_speech.api.json` synthesizes text, `save_voice.api.json` encodes and saves a reference, and `saved_voice_tts.api.json` loads a saved voice. Replace the example audio filename and transcript with matching local data. A saved-voice graph has completed on the local ComfyUI server and saved FLAC through Save Audio. A second request reused the cached loader and voice nodes. An actual server restart remains unverified.

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

Enable `comfy_ops=True`, `cuda_graphs=True` and `attention_backend=flash` for the fastest measured setting here; choose `comfy_kitchen_int8` to use Kitchen attention. Defaults remain unchanged, and outputs can differ numerically between backends. Model/codec loading and audio saving are outside these generation timings.

`benchmarks/native_ops.py --output <directory>` compares prefill results and four variants (original/native, with/without CUDA Graphs), writes timings to JSON and produces comparison WAV files. It loads only local model and voice files.

`benchmarks/compare_attention.py --profile --output <directory>` performs the interleaved attention comparison and records CUDA kernel evidence. `benchmarks/attention_backends.py` measures the registered implementations at representative Audio8 attention shapes.

After restarting ComfyUI, enable the optional `cuda_graphs` input on Audio8 TTS to replay its decoder with CUDA Graphs. It defaults to off and is ignored on CPU. `cuda_graph_scope` selects the extent: `fast` (default) replays only the fast decoder, `frame` replays the whole frame, so one frame becomes two graph replays. The patch uses ComfyUI's ModelPatcher, preserves sampling, and releases graph tensors after each generation attempt; retries and chunks receive fresh graphs. It requires no Torch update or additional dependencies. Capture has an upfront cost, so very short outputs can benefit less.

On the local RTX 4090 / Torch 2.8.0+cu128 / BF16 setup, a 75-frame utterance took a median 5.00 seconds with the standard path and 2.95 seconds with CUDA Graphs (two runs each, model already loaded, graph capture included). Generated codes were exactly equal. These are generation timings; initial model/codec loading and audio saving are separate.

Measured again on the same setup with 96 tokens and greedy decoding, so all three variants are directly comparable: 17.21 s without graphs, 8.28 s with `cuda_graph_scope=fast`, 2.82 s with `frame`. Generated codes stayed exactly equal to the ungraphed path. With the node default sampling the same comparison was 8.64 s without graphs against 1.99 s with `frame`. Sampling draws its random numbers inside the graph, so one seed yields other codes than the ungraphed path while a different seed still changes the output. `frame` reserves more graph memory per generation attempt.

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

See [PROGRESS.md](PROGRESS.md) for the current state and [InplementationPlan.md](InplementationPlan.md) for remaining work.
