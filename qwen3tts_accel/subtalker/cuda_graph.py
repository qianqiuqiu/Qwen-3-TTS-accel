"""CUDA Graph sub-talker runner for Qwen3-TTS code_predictor.

Replaces the stock HF sub-talker generation path with pre-captured CUDA
graphs, eliminating virtually all CPU overhead from the 15-step
sub-talker decode loop.

Architecture:
  - code_predictor has 5 transformer layers
  - Prefill: [B, 2, hidden] (past_hidden + codec_0_embed, projected)
  - Decode: 14 steps of [B, 1, hidden], one per codebook 1-14
  - Each step uses a separate lm_head[i] and codec_embedding[i]
  - Total KV positions: 2 (prefill) + 14 (decode) = 16 max

Design for CUDA graphs:
  - Pre-allocated KV cache with fixed max-seq-len dimension
  - Attention always reads the full KV buffer with additive float mask
    to zero out future/unused positions — keeps tensor shapes static
  - Separate graphs for prefill (seq=2) and each decode step (seq=1)
  - KV cache writes use compile-time-known integer slice indices
  - lm_head + sampling stay outside graphs (different head per step +
    non-deterministic multinomial)
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_NUM_CODEBOOKS = 15   # codebooks 1-15
_MAX_SEQ_LEN = 17     # 2 prefill + 15 decode


def _sample_from_logits(
    logits: torch.Tensor,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
) -> torch.Tensor:
    """Sample tokens from logits. Returns [B, 1] tensor."""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = (cum_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        logits = logits.scatter(dim=-1, index=sorted_idx, src=sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. cos/sin: [B, S, D], q/k: [B, H, S, D]."""
    cos = cos.unsqueeze(1)  # [B, 1, S, D]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class CUDAGraphSubTalkerRunner:
    """Sub-talker with CUDA Graph capture for minimal CPU overhead."""

    def __init__(
        self,
        code_predictor: Any,
        max_batch_size: int = 1,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._device = torch.device(device)
        self._dtype = dtype
        self._max_batch = max_batch_size

        self._projection = code_predictor.small_to_mtp_projection
        self._model = code_predictor.model
        self._lm_heads = code_predictor.lm_head
        self._codec_embeddings = self._model.codec_embedding
        self._layers = self._model.layers
        self._norm = self._model.norm
        self._rotary_emb = self._model.rotary_emb
        self._num_layers = len(self._layers)
        self._config = self._model.config

        layer0 = self._layers[0].self_attn
        self._head_dim = layer0.head_dim
        self._num_heads = self._config.num_attention_heads
        self._num_kv_heads = self._config.num_key_value_heads
        self._num_kv_groups = self._num_heads // self._num_kv_heads
        self._hidden_dim = self._config.hidden_size

        self._kv_cache = self._allocate_kv_cache()

        # Pre-built attention masks
        self._prefill_attn_mask = self._build_prefill_mask()
        self._decode_attn_masks = [
            self._build_decode_mask(step) for step in range(_NUM_CODEBOOKS - 1)
        ]

        # Pre-built position_ids tensors (static, safe for CUDA graph capture)
        self._prefill_pos_ids = torch.arange(2, device=self._device).unsqueeze(0).expand(max_batch_size, -1)
        self._decode_pos_ids = [
            torch.tensor([[2 + s]], device=self._device).expand(max_batch_size, -1)
            for s in range(_NUM_CODEBOOKS - 1)
        ]

        # Graph state
        self._prefill_graph: torch.cuda.CUDAGraph | None = None
        self._decode_graphs: list[torch.cuda.CUDAGraph | None] = [None] * (_NUM_CODEBOOKS - 1)
        self._static_prefill_in: torch.Tensor | None = None
        self._static_prefill_out: torch.Tensor | None = None
        self._static_decode_in: torch.Tensor | None = None
        self._static_decode_outs: list[torch.Tensor] = []
        self._captured = False

    def _allocate_kv_cache(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        B, H, S, D = self._max_batch, self._num_kv_heads, _MAX_SEQ_LEN, self._head_dim
        return [
            (torch.zeros(B, H, S, D, device=self._device, dtype=self._dtype),
             torch.zeros(B, H, S, D, device=self._device, dtype=self._dtype))
            for _ in range(self._num_layers)
        ]

    def _zero_kv_cache(self) -> None:
        for k, v in self._kv_cache:
            k.zero_()
            v.zero_()

    def _build_prefill_mask(self) -> torch.Tensor:
        mask = torch.full((1, 1, 2, _MAX_SEQ_LEN), float("-inf"), device=self._device, dtype=self._dtype)
        mask[0, 0, 0, 0] = 0.0
        mask[0, 0, 1, 0:2] = 0.0
        return mask

    def _build_decode_mask(self, step: int) -> torch.Tensor:
        mask = torch.full((1, 1, 1, _MAX_SEQ_LEN), float("-inf"), device=self._device, dtype=self._dtype)
        mask[0, 0, 0, :2 + step + 1] = 0.0
        return mask

    # ------------------------------------------------------------------
    # Transformer step builders — one for prefill, one per decode step
    # Each has fixed integer slice positions, safe for CUDA graph capture.
    # ------------------------------------------------------------------

    def _make_prefill_fn(self):
        """Return a callable that runs prefill (positions 0,1) with fixed KV writes."""
        B = self._max_batch
        layers = self._layers
        norm = self._norm
        rotary = self._rotary_emb
        kv_cache = self._kv_cache
        num_heads = self._num_heads
        num_kv_heads = self._num_kv_heads
        num_kv_groups = self._num_kv_groups
        head_dim = self._head_dim
        mask = self._prefill_attn_mask
        pos_ids = self._prefill_pos_ids

        def prefill_fn(hidden_states: torch.Tensor) -> torch.Tensor:
            cos, sin = rotary(hidden_states, pos_ids)

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                q = attn.q_norm(attn.q_proj(hidden_states).view(*input_shape, num_heads, head_dim)).transpose(1, 2)
                k = attn.k_norm(attn.k_proj(hidden_states).view(*input_shape, num_kv_heads, head_dim)).transpose(1, 2)
                v = attn.v_proj(hidden_states).view(*input_shape, num_kv_heads, head_dim).transpose(1, 2)

                q, k = _apply_rotary_pos_emb(q, k, cos, sin)

                # Write KV at positions 0 and 1 (integer slices, graph-safe)
                kc, vc = kv_cache[layer_idx]
                kc[:B, :, 0:2, :] = k
                vc[:B, :, 0:2, :] = v

                k_full = kc[:B]
                v_full = vc[:B]
                if num_kv_groups > 1:
                    k_full = k_full.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, num_heads, _MAX_SEQ_LEN, head_dim)
                    v_full = v_full.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, num_heads, _MAX_SEQ_LEN, head_dim)

                attn_output = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask, scale=attn.scaling)
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                attn_output = attn.o_proj(attn_output)
                hidden_states = residual + attn_output

                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            return norm(hidden_states)

        return prefill_fn

    def _make_decode_fn(self, step: int):
        """Return a callable for decode step `step` (position = 2 + step)."""
        B = self._max_batch
        pos_int = 2 + step  # integer, baked into the function
        layers = self._layers
        norm = self._norm
        rotary = self._rotary_emb
        kv_cache = self._kv_cache
        num_heads = self._num_heads
        num_kv_heads = self._num_kv_heads
        num_kv_groups = self._num_kv_groups
        head_dim = self._head_dim
        mask = self._decode_attn_masks[step]
        pos_ids = self._decode_pos_ids[step]

        def decode_fn(hidden_states: torch.Tensor) -> torch.Tensor:
            cos, sin = rotary(hidden_states, pos_ids)

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                q = attn.q_norm(attn.q_proj(hidden_states).view(*input_shape, num_heads, head_dim)).transpose(1, 2)
                k = attn.k_norm(attn.k_proj(hidden_states).view(*input_shape, num_kv_heads, head_dim)).transpose(1, 2)
                v = attn.v_proj(hidden_states).view(*input_shape, num_kv_heads, head_dim).transpose(1, 2)

                q, k = _apply_rotary_pos_emb(q, k, cos, sin)

                # Write KV at fixed position (integer, graph-safe)
                kc, vc = kv_cache[layer_idx]
                kc[:B, :, pos_int:pos_int+1, :] = k
                vc[:B, :, pos_int:pos_int+1, :] = v

                k_full = kc[:B]
                v_full = vc[:B]
                if num_kv_groups > 1:
                    k_full = k_full.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, num_heads, _MAX_SEQ_LEN, head_dim)
                    v_full = v_full.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, num_heads, _MAX_SEQ_LEN, head_dim)

                attn_output = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask, scale=attn.scaling)
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                attn_output = attn.o_proj(attn_output)
                hidden_states = residual + attn_output

                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            return norm(hidden_states)

        return decode_fn

    # ------------------------------------------------------------------
    # Graph capture
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def capture_graphs(self) -> None:
        if self._captured:
            return

        B = self._max_batch
        logger.info(
            "Capturing CUDA graphs for sub-talker (B=%d, layers=%d, kv_heads=%d, head_dim=%d)",
            B, self._num_layers, self._num_kv_heads, self._head_dim,
        )

        self._model.eval()
        for h in self._lm_heads:
            h.eval()
        self._projection.eval()

        # Build step functions
        prefill_fn = self._make_prefill_fn()
        decode_fns = [self._make_decode_fn(s) for s in range(_NUM_CODEBOOKS - 1)]

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            # --- Prefill graph ---
            self._static_prefill_in = torch.zeros(
                B, 2, self._hidden_dim, device=self._device, dtype=self._dtype
            )

            # Warmup
            for _ in range(3):
                self._zero_kv_cache()
                _ = prefill_fn(self._static_prefill_in)

            self._zero_kv_cache()
            self._prefill_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._prefill_graph, stream=s):
                self._static_prefill_out = prefill_fn(self._static_prefill_in)

            # --- Decode graphs ---
            self._static_decode_in = torch.zeros(
                B, 1, self._hidden_dim, device=self._device, dtype=self._dtype
            )
            self._static_decode_outs = []

            for step in range(_NUM_CODEBOOKS - 1):
                fn = decode_fns[step]

                # Warmup: populate KV for prior steps
                for _ in range(3):
                    self._zero_kv_cache()
                    _ = prefill_fn(self._static_prefill_in)
                    for prior in range(step):
                        _ = decode_fns[prior](self._static_decode_in)
                    _ = fn(self._static_decode_in)

                # Final KV state before capture
                self._zero_kv_cache()
                _ = prefill_fn(self._static_prefill_in)
                for prior in range(step):
                    _ = decode_fns[prior](self._static_decode_in)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    out = fn(self._static_decode_in)
                self._decode_graphs[step] = graph
                self._static_decode_outs.append(out)

        torch.cuda.current_stream().wait_stream(s)
        self._captured = True

        # Store step functions for eager execution when graph replay cannot be used
        self._prefill_fn = prefill_fn
        self._decode_fns = decode_fns

        logger.info("CUDA graph capture complete for sub-talker (%d decode graphs)", len(self._decode_graphs))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ) -> torch.Tensor:
        """Generate codebook tokens 1-15 given [B, 2, 2048] inputs.

        Returns [B, 15] tensor of codec tokens for codebooks 1-15.
        """
        B = inputs_embeds.shape[0]
        assert B <= self._max_batch, f"Batch {B} > max {self._max_batch}"
        projected = self._projection(inputs_embeds)

        if self._captured and B == self._max_batch:
            return self._generate_with_graphs(projected, B, do_sample, top_k, top_p, temperature)
        return self._generate_eager(projected, B, do_sample, top_k, top_p, temperature)

    def _generate_with_graphs(
        self, projected: torch.Tensor, B: int,
        do_sample: bool, top_k: int, top_p: float, temperature: float,
    ) -> torch.Tensor:
        tokens = []
        self._zero_kv_cache()

        # Prefill
        self._static_prefill_in[:B].copy_(projected)
        self._prefill_graph.replay()
        hidden = self._static_prefill_out

        logits = self._lm_heads[0](hidden[:B, -1, :])
        tok = _sample_from_logits(logits, do_sample, top_k, top_p, temperature)
        tokens.append(tok)

        # Decode steps
        for step in range(_NUM_CODEBOOKS - 1):
            embed = self._codec_embeddings[step](tokens[-1])
            proj = self._projection(embed)
            self._static_decode_in[:B].copy_(proj)
            self._decode_graphs[step].replay()
            hidden = self._static_decode_outs[step]

            logits = self._lm_heads[step + 1](hidden[:B, -1, :])
            tok = _sample_from_logits(logits, do_sample, top_k, top_p, temperature)
            tokens.append(tok)

        return torch.cat(tokens, dim=-1)

    def _generate_eager(
        self, projected: torch.Tensor, B: int,
        do_sample: bool, top_k: int, top_p: float, temperature: float,
    ) -> torch.Tensor:
        tokens = []
        self._zero_kv_cache()

        # Use stored functions if available, else build on the fly
        prefill_fn = getattr(self, '_prefill_fn', None) or self._make_prefill_fn()
        decode_fns = getattr(self, '_decode_fns', None) or [self._make_decode_fn(s) for s in range(_NUM_CODEBOOKS - 1)]

        hidden = prefill_fn(projected)
        logits = self._lm_heads[0](hidden[:B, -1, :])
        tok = _sample_from_logits(logits, do_sample, top_k, top_p, temperature)
        tokens.append(tok)

        for step in range(_NUM_CODEBOOKS - 1):
            embed = self._codec_embeddings[step](tokens[-1])
            proj = self._projection(embed)
            hidden = decode_fns[step](proj)
            logits = self._lm_heads[step + 1](hidden[:B, -1, :])
            tok = _sample_from_logits(logits, do_sample, top_k, top_p, temperature)
            tokens.append(tok)

        return torch.cat(tokens, dim=-1)


# ------------------------------------------------------------------
# Drop-in result wrapper
# ------------------------------------------------------------------

class _CUDAGraphResult:
    __slots__ = ("sequences",)
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences


def patch_code_predictor_cuda_graph(
    talker: Any,
    max_batch_size: int = 1,
    capture: bool = True,
) -> CUDAGraphSubTalkerRunner:
    """Patch talker.code_predictor.generate() to use CUDA graph runner."""
    cp = talker.code_predictor
    device = next(cp.parameters()).device
    dtype = next(cp.parameters()).dtype

    # Switch to SDPA
    for layer in cp.model.layers:
        attn = layer.self_attn
        attn.config._attn_implementation = "sdpa"
        attn._attn_implementation = "sdpa"

    runner = CUDAGraphSubTalkerRunner(
        code_predictor=cp,
        max_batch_size=max_batch_size,
        device=device,
        dtype=dtype,
    )

    if capture:
        runner.capture_graphs()

    cp._original_generate = cp.generate

    def _cuda_graph_generate(
        *,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 15,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        **_kwargs,
    ):
        seqs = runner.generate(
            inputs_embeds,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        return _CUDAGraphResult(seqs)

    cp.generate = _cuda_graph_generate
    return runner
