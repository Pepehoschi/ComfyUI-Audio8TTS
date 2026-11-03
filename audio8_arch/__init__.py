"""Bundled Audio8 (Arktts) model architecture.

The four modules in this package are verbatim copies of the architecture
files shipped in the Edge0/Audio8-TTS-Realistic model repositories on
Hugging Face, which carry the Apache-2.0 license (see LICENSE in the
node root and NOTICE.upstream for the upstream attribution, which is
retained as required by Apache-2.0 section 4).

Source integrity (sha256, captured when bundled from the local model
directory of Audio8-TTS-Realistic-0.6b):

    configuration_arktts.py     27fa4fd4073ef1983a48919a4562300cace84687335035a9b9a077966e1ac04e
    modeling_arktts.py          b320b6c875e6a57afb9745cc98fc72c116d9df6b3bf20d092a5ad3095026911f
    modeling_arktts_codec.py    e1d6b0e1ef1e9ad303dc01d45d86229cf7f2e98d06a7b426234a7e4d7e3ede68
    processing_arktts.py        eba6ecb6b7f0e221bcd19c5e3fe5978e677faf6b362912ff4febc08c3dc7f9df

Usage: audio8_pack.load_pack_model instantiates these classes directly when
loading a single-file pack and fills them through load_state_dict; no
trust_remote_code, dynamic-module resolution or Auto* registration is
involved. Do not register these classes with the transformers Auto*
factories: transformers 5.x then skips materializing the checkpoint weights
and the model silently keeps constructor-initialized values (verified on
transformers 5.9.0).
"""
