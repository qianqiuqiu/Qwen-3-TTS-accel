"""Direct SDPA-based main talker runner for Qwen3-TTS.

Bypasses HF generate() and vLLM AsyncLLMEngine by directly driving the
28-layer main talker transformer with manual KV cache management and SDPA.

Architecture:
  - Reuses the HF model's existing layers, embeddings, and head in-place.
  - Manages a pre-allocated KV cache (like CUDAGraphSubTalkerRunner does for
    the 5-layer sub-talker).
  - During prefill: runs inputs_embeds through all 28 layers with SDPA.
  - During decode: runs sub-talker, computes next embedding, then runs
    one token through the transformer with KV cache.
  - M-RoPE position_ids: for pure-TTS (no vision), all 3 channels share the
    same linear position, so M-RoPE degenerates to standard multi-head RoPE.

This avoids:
  - HF generate() overhead (prepare_inputs, _update_model_kwargs, etc.)
  - DynamicCache overhead (allocation, copying)
  - vLLM engine startup and scheduler complexity
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply M-RoPE (multimodal rotary position embedding).

    For pure-TTS sequences all 3 modality channels have the same position_ids,
    so the interleaved merge just copies the same values.  We still run the
    full M-RoPE code-path for correctness.

    cos, sin: [3, B, S, head_dim]  (from Qwen3TTSTalkerRotaryEmbedding)
    q, k:     [B, num_heads, S, head_dim]
    """
    if interleaved:
        def _apply_interleaved_rope(x, modality_num):
            x_t = x[0].clone()
            for i in range(1, modality_num):
                n = mrope_section[i]
                beg_idx = i
                end_idx = n * modality_num
                x_t[..., beg_idx:end_idx:modality_num] = x[i, ..., beg_idx:end_idx:modality_num]
            return x_t

        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        cos_merged = torch.cat(
            [_apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1
        ).unsqueeze(1)  # [B, 1, S, head_dim]
        sin_merged = torch.cat(
            [_apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1
        ).unsqueeze(1)
    else:
        sec2 = mrope_section * 2
        cos_merged = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(sec2, dim=-1))], dim=-1
        ).unsqueeze(1)
        sin_merged = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(sec2, dim=-1))], dim=-1
        ).unsqueeze(1)

    q_embed = (q * cos_merged) + (_rotate_half(q) * sin_merged)
    k_embed = (k * cos_merged) + (_rotate_half(k) * sin_merged)
    return q_embed, k_embed


def _sample_from_logits(
    logits: torch.Tensor,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float = 1.0,
    past_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a single token from logits. Returns [B, 1]."""
    if repetition_penalty != 1.0 and past_tokens is not None and past_tokens.numel() > 0:
        for b in range(logits.shape[0]):
            unique_past = past_tokens[b].unique()
            score = logits[b, unique_past]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits[b, unique_past] = score

    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)

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


class MainTalkerRunner:
    """Direct SDPA-based inference runner for the 28-layer main talker.

    Wraps the HF model's existing layers (no weight copying) and manages
    a static KV cache for efficient prefill + decode.
    """

    def __init__(
        self,
        talker: Any,
        *,
        max_seq_len: int = 4096,
        max_batch_size: int = 1,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._device = torch.device(device)
        self._dtype = dtype
        self._max_batch = max_batch_size
        self._max_seq_len = max_seq_len

        # Reference HF model components (no copying)
        self._talker = talker
        self._model = talker.model  # Qwen3TTSTalkerModel
        self._layers = self._model.layers
        self._norm = self._model.norm
        self._rotary_emb = self._model.rotary_emb
        self._codec_embedding = self._model.codec_embedding
        self._text_embedding = self._model.text_embedding
        self._codec_head = talker.codec_head
        self._text_projection = talker.text_projection
        self._code_predictor = talker.code_predictor
        self._config = self._model.config

        self._num_layers = len(self._layers)
        layer0_attn = self._layers[0].self_attn
        self._head_dim = layer0_attn.head_dim
        self._num_heads = self._config.num_attention_heads
        self._num_kv_heads = self._config.num_key_value_heads
        self._num_kv_groups = self._num_heads // self._num_kv_heads
        self._hidden_dim = self._config.hidden_size

        # M-RoPE config
        rope_scaling = self._config.rope_scaling or {}
        self._mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])
        self._mrope_interleaved = rope_scaling.get("interleaved", True)

        # Allocate KV cache: [B, num_kv_heads, max_seq_len, head_dim]
        self._kv_cache = self._allocate_kv_cache()

        # EOS token
        self._eos_token_id = self._config.codec_eos_token_id

        # Sub-talker codec embeddings (from code_predictor)
        self._subtalker_codec_embeddings = self._code_predictor.model.codec_embedding

        logger.info(
            "MainTalkerRunner initialized: %d layers, %d heads (%d kv), "
            "head_dim=%d, hidden=%d, max_seq=%d",
            self._num_layers, self._num_heads, self._num_kv_heads,
            self._head_dim, self._hidden_dim, self._max_seq_len,
        )

    def _allocate_kv_cache(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        B = self._max_batch
        H = self._num_kv_heads
        S = self._max_seq_len
        D = self._head_dim
        return [
            (
                torch.zeros(B, H, S, D, device=self._device, dtype=self._dtype),
                torch.zeros(B, H, S, D, device=self._device, dtype=self._dtype),
            )
            for _ in range(self._num_layers)
        ]

    def _zero_kv_cache(self) -> None:
        for k, v in self._kv_cache:
            k.zero_()
            v.zero_()

    def _build_causal_mask(
        self, query_len: int, kv_len: int
    ) -> torch.Tensor:
        """Build additive causal attention mask.

        Returns [1, 1, query_len, max_seq_len] with -inf for masked positions.
        """
        mask = torch.full(
            (1, 1, query_len, self._max_seq_len),
            float("-inf"),
            device=self._device,
            dtype=self._dtype,
        )
        # Allow attending to positions [0, kv_len) with causal constraint
        for q in range(query_len):
            # Position in KV of this query token: kv_len - query_len + q
            kv_pos = kv_len - query_len + q
            mask[0, 0, q, : kv_pos + 1] = 0.0
        return mask

    def _build_position_ids(
        self, start: int, length: int, batch_size: int
    ) -> torch.Tensor:
        """Build M-RoPE position_ids [3, B, length]."""
        pos = torch.arange(start, start + length, device=self._device)
        # For pure TTS, all 3 channels share the same positions
        pos = pos.unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
        return pos

    def _transformer_forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_write_start: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Run the 28-layer transformer with SDPA and manual KV cache.

        Args:
            hidden_states: [B, S, D]
            attn_mask: [1, 1, S, max_seq_len]
            position_ids: [3, B, S]
            kv_write_start: starting index in KV cache to write new KV
            seq_len: length of the input sequence (S)
        """
        B = hidden_states.shape[0]

        # Compute rotary embeddings
        cos, sin = self._rotary_emb(hidden_states, position_ids)
        # cos, sin: [3, B, S, head_dim]

        for layer_idx, layer in enumerate(self._layers):
            attn = layer.self_attn
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]  # [B, S]
            hidden_shape = (*input_shape, -1, self._head_dim)

            # QKV projections
            q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            # q: [B, num_heads, S, head_dim], k/v: [B, num_kv_heads, S, head_dim]

            # Apply M-RoPE
            q, k = _apply_mrope(
                q, k, cos, sin,
                self._mrope_section, self._mrope_interleaved,
            )

            # Write KV to cache
            kc, vc = self._kv_cache[layer_idx]
            kc[:B, :, kv_write_start:kv_write_start + seq_len, :] = k
            vc[:B, :, kv_write_start:kv_write_start + seq_len, :] = v

            # Read full KV and expand for GQA
            k_full = kc[:B]
            v_full = vc[:B]
            if self._num_kv_groups > 1:
                k_full = k_full.unsqueeze(2).expand(
                    -1, -1, self._num_kv_groups, -1, -1
                ).reshape(B, self._num_heads, self._max_seq_len, self._head_dim)
                v_full = v_full.unsqueeze(2).expand(
                    -1, -1, self._num_kv_groups, -1, -1
                ).reshape(B, self._num_heads, self._max_seq_len, self._head_dim)

            # SDPA with additive mask
            attn_output = F.scaled_dot_product_attention(
                q, k_full, v_full,
                attn_mask=attn_mask,
                scale=attn.scaling,
            )
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            # MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self._norm(hidden_states)
        return hidden_states

    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        *,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        subtalker_do_sample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
    ) -> tuple[list[list[int]], torch.Tensor]:
        """Run prefill + decode loop for the main talker.

        Args:
            inputs_embeds: [B, L, D] - prefill embeddings from preprocessor
            attention_mask: [B, L] - attention mask (all ones for single request)
            trailing_text_hidden: [B, T, D] - remaining text embeddings
            tts_pad_embed: [B, 1, D] - padding embedding
            max_new_tokens: max codec_0 tokens to generate
            min_new_tokens: min before EOS is allowed

        Returns:
            (codec_frames, last_hidden_state):
                codec_frames: list of [16] codec token lists per frame
                last_hidden_state: [B, 1, D] final hidden state
        """
        B = inputs_embeds.shape[0]
        prefill_len = inputs_embeds.shape[1]
        assert B <= self._max_batch, f"Batch {B} > max {self._max_batch}"
        assert prefill_len <= self._max_seq_len, (
            f"Prefill length {prefill_len} > max {self._max_seq_len}"
        )

        self._zero_kv_cache()

        # Ensure SDPA attention for all layers
        for layer in self._layers:
            layer.self_attn.config._attn_implementation = "sdpa"
            layer.self_attn._attn_implementation = "sdpa"

        # ---- PREFILL ----
        position_ids = self._build_position_ids(0, prefill_len, B)
        attn_mask = self._build_causal_mask(prefill_len, prefill_len)

        hidden = self._transformer_forward(
            inputs_embeds, attn_mask, position_ids,
            kv_write_start=0, seq_len=prefill_len,
        )
        # hidden: [B, prefill_len, D]

        # Get logits for last token
        logits = self._codec_head(hidden[:, -1:, :]).squeeze(1)  # [B, vocab_size]
        past_tokens_list = []

        # First token
        tok = _sample_from_logits(
            logits, do_sample, top_k, top_p, temperature,
        )  # [B, 1]
        past_tokens_list.append(tok)

        # Track last hidden for sub-talker
        last_hidden = hidden[:, -1:, :]  # [B, 1, D]

        codec_frames: list[list[int]] = []
        kv_len = prefill_len
        generation_step = 0

        # ---- DECODE LOOP ----
        for step in range(max_new_tokens):
            codec_0_token = tok[0, 0].item()

            # Check EOS (only after min_new_tokens)
            if step >= min_new_tokens and codec_0_token == self._eos_token_id:
                break

            # Run sub-talker to get codebooks 1-15
            codec_0_embed = self._codec_embedding(tok)  # [B, 1, D]
            sub_input = torch.cat([last_hidden, codec_0_embed], dim=1)  # [B, 2, D]

            predictor_result = self._code_predictor.generate(
                inputs_embeds=sub_input,
                max_new_tokens=self._config.num_code_groups - 1,
                do_sample=subtalker_do_sample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
            )
            if hasattr(predictor_result, 'sequences'):
                fine_tokens = predictor_result.sequences  # [B, 15]
            else:
                fine_tokens = predictor_result  # [B, 15]

            # Build full codec frame
            frame = [codec_0_token] + fine_tokens[0].tolist()
            codec_frames.append(frame)

            # Compute next input embedding: sum of all 16 codec embeddings
            codec_hiddens = [codec_0_embed]  # [B, 1, D]
            for i in range(self._config.num_code_groups - 1):
                emb = self._subtalker_codec_embeddings[i](
                    fine_tokens[:, i:i + 1]
                )  # [B, 1, D]
                codec_hiddens.append(emb)

            # Sum all codec embeddings -> [B, 1, D]
            all_embeds = torch.cat(codec_hiddens, dim=1)  # [B, 16, D]
            next_input = all_embeds.sum(1, keepdim=True)  # [B, 1, D]

            # Add text conditioning
            if generation_step < trailing_text_hidden.shape[1]:
                next_input = next_input + trailing_text_hidden[:, generation_step:generation_step + 1]
            else:
                next_input = next_input + tts_pad_embed

            generation_step += 1

            # Run one decode step through transformer
            kv_len += 1
            if kv_len > self._max_seq_len:
                logger.warning("KV cache overflow at step %d (kv_len=%d)", step, kv_len)
                break

            decode_pos = self._build_position_ids(kv_len - 1, 1, B)
            decode_mask = self._build_causal_mask(1, kv_len)

            hidden = self._transformer_forward(
                next_input, decode_mask, decode_pos,
                kv_write_start=kv_len - 1, seq_len=1,
            )
            # hidden: [B, 1, D]
            last_hidden = hidden

            logits = self._codec_head(hidden).squeeze(1)  # [B, vocab_size]

            past_tok_tensor = torch.cat(past_tokens_list, dim=-1) if past_tokens_list else None
            tok = _sample_from_logits(
                logits, do_sample, top_k, top_p, temperature,
                repetition_penalty=repetition_penalty,
                past_tokens=past_tok_tensor,
            )
            past_tokens_list.append(tok)

        return codec_frames, last_hidden

    @torch.inference_mode()
    def generate_streaming(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        trailing_text_hidden: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        *,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        subtalker_do_sample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
    ):
        """Streaming version: yields codec frames one at a time.

        Yields:
            list[int]: a 16-element codec frame [codec_0, codec_1, ..., codec_15]
        """
        B = inputs_embeds.shape[0]
        prefill_len = inputs_embeds.shape[1]
        assert B <= self._max_batch
        assert prefill_len <= self._max_seq_len

        self._zero_kv_cache()

        for layer in self._layers:
            layer.self_attn.config._attn_implementation = "sdpa"
            layer.self_attn._attn_implementation = "sdpa"

        # ---- PREFILL ----
        position_ids = self._build_position_ids(0, prefill_len, B)
        attn_mask = self._build_causal_mask(prefill_len, prefill_len)

        hidden = self._transformer_forward(
            inputs_embeds, attn_mask, position_ids,
            kv_write_start=0, seq_len=prefill_len,
        )

        logits = self._codec_head(hidden[:, -1:, :]).squeeze(1)
        past_tokens_list = []

        tok = _sample_from_logits(logits, do_sample, top_k, top_p, temperature)
        past_tokens_list.append(tok)

        last_hidden = hidden[:, -1:, :]
        kv_len = prefill_len
        generation_step = 0

        # ---- DECODE LOOP ----
        for step in range(max_new_tokens):
            codec_0_token = tok[0, 0].item()

            if step >= min_new_tokens and codec_0_token == self._eos_token_id:
                break

            # Sub-talker
            codec_0_embed = self._codec_embedding(tok)
            sub_input = torch.cat([last_hidden, codec_0_embed], dim=1)

            predictor_result = self._code_predictor.generate(
                inputs_embeds=sub_input,
                max_new_tokens=self._config.num_code_groups - 1,
                do_sample=subtalker_do_sample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
            )
            if hasattr(predictor_result, 'sequences'):
                fine_tokens = predictor_result.sequences
            else:
                fine_tokens = predictor_result

            frame = [codec_0_token] + fine_tokens[0].tolist()
            yield frame

            # Next input embedding
            codec_hiddens = [codec_0_embed]
            for i in range(self._config.num_code_groups - 1):
                emb = self._subtalker_codec_embeddings[i](fine_tokens[:, i:i + 1])
                codec_hiddens.append(emb)

            all_embeds = torch.cat(codec_hiddens, dim=1)
            next_input = all_embeds.sum(1, keepdim=True)

            if generation_step < trailing_text_hidden.shape[1]:
                next_input = next_input + trailing_text_hidden[:, generation_step:generation_step + 1]
            else:
                next_input = next_input + tts_pad_embed

            generation_step += 1

            kv_len += 1
            if kv_len > self._max_seq_len:
                logger.warning("KV cache overflow at step %d", step)
                break

            decode_pos = self._build_position_ids(kv_len - 1, 1, B)
            decode_mask = self._build_causal_mask(1, kv_len)

            hidden = self._transformer_forward(
                next_input, decode_mask, decode_pos,
                kv_write_start=kv_len - 1, seq_len=1,
            )
            last_hidden = hidden

            logits = self._codec_head(hidden).squeeze(1)
            past_tok_tensor = torch.cat(past_tokens_list, dim=-1) if past_tokens_list else None
            tok = _sample_from_logits(
                logits, do_sample, top_k, top_p, temperature,
                repetition_penalty=repetition_penalty,
                past_tokens=past_tok_tensor,
            )
            past_tokens_list.append(tok)


def create_main_talker_runner(
    model: Any,
    *,
    max_seq_len: int = 4096,
    max_batch_size: int = 1,
    apply_cuda_graph_subtalker: bool = True,
) -> MainTalkerRunner:
    """Create a MainTalkerRunner from a Qwen3TTS model.

    Args:
        model: Qwen3TTSModel (inference wrapper) or Qwen3TTSForConditionalGeneration
        max_seq_len: max sequence length for KV cache
        max_batch_size: max batch size
        apply_cuda_graph_subtalker: whether to apply CUDA graph optimization
            to the sub-talker (code_predictor) before creating the runner
    """
    import torch as _torch

    if hasattr(model, "model") and hasattr(model, "processor"):
        hf_model = model.model  # Qwen3TTSForConditionalGeneration
    else:
        hf_model = model

    talker = hf_model.talker
    device = next(talker.parameters()).device
    dtype = next(talker.parameters()).dtype

    if apply_cuda_graph_subtalker and _torch.cuda.is_available():
        from ..subtalker.cuda_graph import patch_code_predictor_cuda_graph
        patch_code_predictor_cuda_graph(talker, max_batch_size=max_batch_size, capture=True)

    runner = MainTalkerRunner(
        talker,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        dtype=dtype,
    )
    return runner
