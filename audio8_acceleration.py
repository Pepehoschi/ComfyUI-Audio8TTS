from contextlib import contextmanager

import torch
import comfy.model_patcher


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
