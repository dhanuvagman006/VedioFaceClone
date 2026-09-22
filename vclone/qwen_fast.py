"""Speed-ups for Qwen3-TTS on small GPUs.

For every audio frame (12.5 per second of speech) the talker asks its 5-layer code predictor for
15 more codebook tokens through a Hugging Face generate() call. On a laptop GPU that is thousands
of tiny kernel launches per sentence, so the GPU mostly waits on Python. This module replaces that
call with (a) CUDA graphs that replay each prediction step with one launch, or (b) if graphs are
unavailable, a lean loop. Both sample exactly like transformers (temperature -> top-k -> top-p)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from transformers import DynamicCache, StaticCache


def _sample(logits: torch.Tensor, do_sample: bool, top_k, top_p, temperature) -> torch.Tensor:
    logits = logits.float()
    if not do_sample:
        return logits.argmax(dim=-1)
    if temperature is not None and temperature != 1.0:
        logits = logits / temperature
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=False)
        remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) <= (1 - top_p)
        remove[..., -1:] = False
        logits = logits.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)


class _GraphedSteps:
    """Captured prediction steps for one batch size: step 0 is the 2-token prefill."""

    def __init__(self, cp, batch: int, in_dim: int):
        self.cp = cp
        dev, dtype = cp.lm_head[0].weight.device, cp.lm_head[0].weight.dtype
        self.n = cp.config.num_code_groups - 1
        length = self.n + 1
        self.cache = StaticCache(config=cp.config, max_cache_len=length)
        self.inp = torch.zeros(batch, 2, in_dim, device=dev, dtype=dtype)
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        causal = torch.ones(length, length, dtype=torch.bool, device=dev).tril()
        self.pos = [torch.arange(2, device=dev)] + [torch.tensor([k + 1], device=dev) for k in range(1, self.n)]
        self.mask = [causal[p][None, None].expand(batch, 1, -1, -1) for p in self.pos]
        self.graphs, self.logits = [], [None] * self.n

        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(stream):  # warm-up allocates the cache and picks kernels
            for _ in range(2):
                for k in range(self.n):
                    self._step(k)
        torch.cuda.current_stream(dev).wait_stream(stream)
        pool = torch.cuda.graph_pool_handle()
        for k in range(self.n):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.logits[k] = self._step(k)
            self.graphs.append(graph)

    def _step(self, k: int) -> torch.Tensor:
        cp = self.cp
        if k == 0:
            emb = cp.small_to_mtp_projection(self.inp)
        else:
            emb = cp.small_to_mtp_projection(cp.model.get_input_embeddings()[k - 1](self.tok))
        out = cp.model(inputs_embeds=emb, attention_mask={"full_attention": self.mask[k]},
                       position_ids=self.pos[k][None], cache_position=self.pos[k],
                       past_key_values=self.cache, use_cache=True)
        return cp.lm_head[k](out.last_hidden_state[:, -1])

    def run(self, inputs_embeds, sample) -> torch.Tensor:
        self.inp.copy_(inputs_embeds)
        tokens = []
        for k, graph in enumerate(self.graphs):
            if k:
                self.tok.copy_(tokens[-1][:, None])
            graph.replay()
            tokens.append(sample(self.logits[k]))
        return torch.stack(tokens, dim=1)


def patch_code_predictor(qwen_model, use_graphs: bool | None = None) -> str:
    """Patch a qwen_tts.Qwen3TTSModel in place. Returns the mode used: 'cuda-graphs' or 'lean'."""
    cp = qwen_model.model.talker.code_predictor
    backbone, embeddings, project = cp.model, cp.model.get_input_embeddings(), cp.small_to_mtp_projection
    if use_graphs is None:
        use_graphs = torch.cuda.is_available() and os.environ.get("VCLONE_NO_CUDA_GRAPHS") != "1"
    graph_sets: dict[int, _GraphedSteps] = {}
    state = {"graphs": use_graphs and cp.lm_head[0].weight.is_cuda}

    @torch.inference_mode()
    def lean(inputs_embeds, max_new_tokens, sample):
        cache = DynamicCache()
        h = backbone(inputs_embeds=project(inputs_embeds), past_key_values=cache,
                     use_cache=True).last_hidden_state[:, -1]
        tokens = []
        for step in range(max_new_tokens):
            tokens.append(sample(cp.lm_head[step](h)))
            if step + 1 < max_new_tokens:
                h = backbone(inputs_embeds=project(embeddings[step](tokens[-1][:, None])), past_key_values=cache,
                             use_cache=True).last_hidden_state[:, -1]
        return torch.stack(tokens, dim=1)

    @torch.inference_mode()
    def generate(inputs_embeds, max_new_tokens, do_sample=True, top_p=1.0, top_k=50, temperature=1.0, **_):
        def sample(logits):
            return _sample(logits, do_sample, top_k, top_p, temperature)

        batch = inputs_embeds.shape[0]
        if state["graphs"] and max_new_tokens == cp.config.num_code_groups - 1 and inputs_embeds.shape[1] == 2:
            try:
                if batch not in graph_sets:
                    graph_sets[batch] = _GraphedSteps(cp, batch, inputs_embeds.shape[-1])
                return SimpleNamespace(sequences=graph_sets[batch].run(inputs_embeds, sample))
            except Exception as exc:  # fall back for good if capture is not possible here
                print(f"[tts] CUDA graphs unavailable ({type(exc).__name__}: {exc}); using the slower path")
                state["graphs"] = False
                graph_sets.clear()
        return SimpleNamespace(sequences=lean(inputs_embeds, max_new_tokens, sample))

    cp.generate = generate
    return "cuda-graphs" if state["graphs"] else "lean"
