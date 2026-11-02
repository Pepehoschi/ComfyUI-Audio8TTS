import logging
from contextlib import contextmanager

import torch
import comfy.model_patcher


logger = logging.getLogger(__name__)

GRAPH_SCOPES = ("fast", "frame")


class _FastDecoderGraph:
    def __init__(self, model):
        self.model = model
        self.captured = None

    def step(self, hidden, position):
        if self.captured is None:
            model = self.model
            static_input = hidden.clone()
            cache_position = torch.zeros(1, device=hidden.device, dtype=torch.long)
            key_mask = torch.ones((hidden.shape[0], model.config.num_codebooks),
                                  device=hidden.device, dtype=torch.bool)

            def compute(value):
                rope = model.fast_freqs_cis[cache_position]
                mask = model._causal_mask(key_mask, cache_position, model.config.num_codebooks)
                for layer in model.fast_layers:
                    value = layer(value, rope, mask, cache_position)
                return model.fast_output(model.fast_norm(value))[:, -1]

            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    compute(static_input)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = compute(static_input)
            # Captured metadata must stay alive for every replay in this generate call.
            self.captured = (graph, static_input, output, cache_position, key_mask)
        graph, static_input, output, cache_position, _ = self.captured
        static_input.copy_(hidden)
        cache_position.fill_(position)
        graph.replay()
        return output


class _SlowDecodeGraph:
    """Replays one single-token slow decode step from a CUDA graph.

    The patched seam also carries the prefill call, whose query length equals the prompt
    width. Only the decode step repeats with constant shapes, so prefill is delegated to
    the unpatched implementation and capture happens on the first decode step. The step
    implementation itself is captured as it currently stands, so a Comfy-attention patch
    with a trimmed key length and the upstream step with its padded mask both work.
    """

    def __init__(self, model, original, max_new_tokens):
        self.model = model
        self.original = original
        self.max_new_tokens = max_new_tokens
        self.captured = None

    def _capture(self, input_ids, cache_position, position_ids, attention_mask):
        model = self.model
        device = input_ids.device
        # The first decode step carries prompt_width + 1 mask entries; the last step of
        # this call reaches prompt_width + max_new_tokens - 1.
        key_width = min(attention_mask.shape[-1] + self.max_new_tokens - 2,
                        model.config.max_seq_len)
        statics = (input_ids.clone(), cache_position.clone(), position_ids.clone(),
                   torch.ones((input_ids.shape[0], key_width), dtype=attention_mask.dtype,
                              device=device))

        def compute():
            return self.original(*statics)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                compute()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = compute()
        # Statics and outputs must stay allocated for every replay in this generate call.
        self.captured = (graph, statics, outputs)
        return self.captured

    def step(self, input_ids, cache_position, position_ids, attention_mask):
        if input_ids.shape[-1] != 1:
            return self.original(input_ids, cache_position, position_ids, attention_mask)
        captured = self.captured
        if captured is None:
            if not bool(attention_mask.all()):
                logger.info("[Audio8] Frame graph kept eager: padded batch")
                return self.original(input_ids, cache_position, position_ids, attention_mask)
            captured = self._capture(input_ids, cache_position, position_ids, attention_mask)
        graph, (ids, cache, positions, _), outputs = captured
        ids.copy_(input_ids)
        cache.copy_(cache_position)
        positions.copy_(position_ids)
        graph.replay()
        return outputs


class _CodebookChainGraph:
    """Replays a whole codebook chain (ten sequential fast steps) with one CUDA graph.

    Upstream builds two small tensors per step, the temperature divisor and the top-p
    threshold inside its logits processor. Both are host to device copies and therefore
    illegal during capture, so this graph inlines the same filtering and keeps those
    constants as static tensors that the warmup pass creates once.
    """

    def __init__(self, model, original):
        self.model = model
        self.original = original
        self.captured = None
        self.constants = None
        self.notified_eager = False

    def _fast_step(self, hidden, cache_position, key_mask):
        model = self.model
        rope = model.fast_freqs_cis[cache_position]
        mask = model._causal_mask(key_mask, cache_position, model.config.num_codebooks)
        for layer in model.fast_layers:
            hidden = layer(hidden, rope, mask, cache_position)
        return model.fast_output(model.fast_norm(hidden))[:, -1]

    def _filter_scores(self, scores, top_k, top_p, temperature):
        constants = self.constants
        if constants is None:
            constants = self.constants = {
                "top_p": torch.tensor(top_p, dtype=scores.dtype, device=scores.device),
                "temperature": torch.tensor(temperature, dtype=scores.dtype,
                                            device=scores.device).clamp_min(1e-5),
                "vocabulary": torch.arange(scores.shape[-1], device=scores.device),
            }
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
        remove_sorted = (cumulative > constants["top_p"]) | (constants["vocabulary"] >= top_k)
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
        return scores.masked_fill(remove, float("-inf")) / constants["temperature"]

    def _chain(self, slow_hidden, semantic, positions, key_mask, top_k, top_p, temperature,
               do_sample, generator):
        model = self.model
        hidden = model.fast_project_in(slow_hidden)
        # Upstream primes codebook position 0 and discards the score. Without that write
        # every replay would read zeros from cache slot 0.
        self._fast_step(hidden, positions[0:1], key_mask)
        current = (semantic - model.config.semantic_begin_id).clamp(
            0, model.config.codebook_size - 1)
        codebooks = [current]
        history = current[:, None]
        hidden = model.fast_embeddings(current)[:, None]
        for position in range(1, model.config.num_codebooks):
            scores = self._fast_step(hidden, positions[position:position + 1], key_mask)
            scores = self._filter_scores(scores, top_k, top_p, temperature)
            current = model._sample(scores, generator=generator) if do_sample \
                else scores.argmax(dim=-1)
            codebooks.append(current)
            history = torch.cat((history, current[:, None]), dim=1)
            hidden = model.fast_embeddings(current)[:, None]
        return torch.stack(codebooks, dim=1)

    def _capture(self, slow_hidden, semantic, top_k, top_p, temperature, do_sample, generator):
        model = self.model
        device = slow_hidden.device
        statics = (slow_hidden.clone(), semantic.clone(),
                   torch.arange(model.config.num_codebooks, device=device, dtype=torch.long),
                   torch.ones((slow_hidden.shape[0], model.config.num_codebooks),
                              device=device, dtype=torch.bool))

        def compute():
            return self._chain(statics[0], statics[1], statics[2], statics[3], top_k, top_p,
                               temperature, do_sample, generator)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                compute()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        if do_sample and generator is not None:
            # The chain samples inside the graph, so its generator has to be part of it.
            graph.register_generator_state(generator)
        with torch.cuda.graph(graph):
            outputs = compute()
        self.captured = (graph, statics, outputs)
        return self.captured

    def chain(self, slow_hidden, semantic, processors, top_k, top_p, temperature, do_sample,
              generator=None):
        if len(processors) > 0:
            if not self.notified_eager:
                logger.info("[Audio8] Frame graph keeps the codebook chain eager: custom "
                            "logits processors may synchronise the host")
                self.notified_eager = True
            return self.original(slow_hidden, semantic, processors, top_k, top_p, temperature,
                                 do_sample, generator)
        captured = self.captured
        if captured is None:
            captured = self._capture(slow_hidden, semantic, top_k, top_p, temperature,
                                     do_sample, generator)
        graph, (hidden, codes, positions, key_mask), outputs = captured
        hidden.copy_(slow_hidden)
        codes.copy_(semantic)
        graph.replay()
        return outputs


@contextmanager
def cuda_graph_fast_decode(bundle):
    patcher = comfy.model_patcher.ModelPatcher(bundle.model, bundle.device, torch.device("cpu"))
    graph = _FastDecoderGraph(bundle.model)
    patcher.add_object_patch("_fast_step", graph.step)
    try:
        patcher.patch_model(load_weights=False)
        yield
    finally:
        patcher.unpatch_model(unpatch_weights=False)
        patcher.object_patches.clear()
        graph.captured = None


@contextmanager
def cuda_graph_frame_decode(bundle, max_new_tokens):
    """Replays both decode seams per frame: slow step plus the whole codebook chain.

    Graph pools are created per generate call and released on exit. Greedy decoding
    reproduces the eager code sequence exactly; sampling draws its random numbers from
    the generator inside the graph, so one seed yields other codes than the eager path.

    The slow-step graph pads its static attention mask to the widest key length of this
    call. That is only safe while the causal mask is still applied. A Comfy attention
    backend that masks single-token decode by trimming keys would instead expose future
    cache slots, which measurably corrupts audio, so that combination falls back.
    """
    if getattr(bundle.model, "audio8_maskless_decode", False):
        logger.warning('[Audio8] cuda_graph_scope="frame" is not combined with a Comfy '
                       "attention backend that drops the decode mask: the padded graph mask "
                       "would reveal future cache slots. Falling back to "
                       'cuda_graph_scope="fast".')
        with cuda_graph_fast_decode(bundle):
            yield
        return

    patcher = comfy.model_patcher.ModelPatcher(bundle.model, bundle.device, torch.device("cpu"))
    slow = _SlowDecodeGraph(bundle.model, bundle.model._slow_step, max_new_tokens)
    chain = _CodebookChainGraph(bundle.model, bundle.model._generate_codebooks)
    patcher.add_object_patch("_slow_step", slow.step)
    patcher.add_object_patch("_generate_codebooks", chain.chain)
    try:
        patcher.patch_model(load_weights=False)
        yield
    finally:
        patcher.unpatch_model(unpatch_weights=False)
        patcher.object_patches.clear()
        slow.captured = None
        chain.captured = None
        chain.constants = None
