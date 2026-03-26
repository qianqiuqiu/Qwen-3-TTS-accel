"""vLLM model plugin for Qwen3-TTS main talker (28-layer transformer).

Registered as ``Qwen3TTSForConditionalGeneration`` in vLLM's ModelRegistry so
that the main talker uses PagedAttention + continuous batching while the
sub-talker (code_predictor) runs with CUDA graphs outside vLLM.

Design:
  - Prefill: receives pre-computed ``inputs_embeds`` from the preprocessor.
  - Decode: receives codec_0 token from vLLM's sampler, runs the CUDA graph
    sub-talker to get codebooks 1-15, computes the summed codec embedding +
    text conditioning, then runs the 28-layer transformer to produce next
    codec_0 logits.
  - Per-sequence state (trailing_text_hidden, tts_pad_embed, past_hidden,
    generation_step) is tracked in SequenceStateStore.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from ..preprocess.preprocessor import PrefillPayload
from ..state.sequence_state import SequenceStateStore, get_default_sequence_store

logger = logging.getLogger(__name__)


class Qwen3TTSTalkerAttention(nn.Module):
    """Single attention layer for the main talker, using vLLM's Attention."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        cache_config: Any = None,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from vllm.model_executor.layers.linear import (
            QKVParallelLinear,
            RowParallelLinear,
        )
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.rotary_embedding import get_rope

        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # QKV merged projection
        attention_bias = getattr(config, "attention_bias", False)
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # QK normalization (Qwen3 style)
        self.q_norm = RMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.k_norm = RMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))

        # M-RoPE for the talker
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])
        interleaved = rope_scaling.get("interleaved", True)
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        max_pos = getattr(config, "max_position_embeddings", 32768)

        # Build rope_parameters dict for get_rope
        rope_params = {
            "rope_type": "mrope_interleaved" if interleaved else "mrope",
            "factor": 1.0,
            "mrope_section": mrope_section,
        }

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_pos,
            is_neox_style=True,
            rope_parameters={
                "rope_type": "mrope_interleaved" if interleaved else "mrope",
                "factor": 1.0,
                "mrope_section": mrope_section,
                "base": rope_theta,
            },
            dtype=torch.get_default_dtype(),
        )

        # vLLM Attention (handles PagedAttention, KV cache, etc.)
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK normalization
        q = q.view(*q.shape[:-1], self.q_size // self.head_dim, self.head_dim)
        q = self.q_norm(q)
        q = q.view(*q.shape[:-2], self.q_size)

        k = k.view(*k.shape[:-1], self.kv_size // self.head_dim, self.head_dim)
        k = self.k_norm(k)
        k = k.view(*k.shape[:-2], self.kv_size)

        # Apply M-RoPE
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3TTSTalkerMLP(nn.Module):
    """MLP for the main talker (SwiGLU / gate-up-down pattern)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        from vllm.model_executor.layers.activation import SiluAndMul

        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3TTSTalkerDecoderLayer(nn.Module):
    """Single decoder layer for the main talker."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        cache_config: Any = None,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm

        self.self_attn = Qwen3TTSTalkerAttention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3TTSTalkerMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3TTSForVLLM(nn.Module):
    """vLLM-compatible model wrapping the Qwen3-TTS 28-layer main talker.

    Responsibilities:
    - Build the transformer backbone with vLLM's PagedAttention layers
    - On prefill: pass through pre-computed inputs_embeds
    - On decode: run sub-talker, compute next input, run transformer
    - Load weights from safetensors with talker.* prefix

    The sub-talker (code_predictor) is loaded separately and managed by the
    CUDA graph runner — it does NOT go through vLLM's model executor.
    """

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        from vllm.config import VllmConfig
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )
        from vllm.model_executor.models.utils import make_layers

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # The HF config is Qwen3TTSConfig. We need the talker sub-config.
        self.config = config
        talker_cfg = config.talker_config
        self.talker_config = talker_cfg

        # Codec embedding: maps codec_0 token -> hidden
        self.codec_embedding = VocabParallelEmbedding(
            talker_cfg.vocab_size,
            talker_cfg.hidden_size,
            quant_config=quant_config,
        )

        # Text embedding (for text_projection, used by preprocessor)
        self.text_embedding = VocabParallelEmbedding(
            talker_cfg.text_vocab_size,
            talker_cfg.text_hidden_size,
            quant_config=quant_config,
        )

        # Transformer layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            talker_cfg.num_hidden_layers,
            lambda pfx: Qwen3TTSTalkerDecoderLayer(
                config=talker_cfg,
                layer_idx=int(pfx.rsplit(".", 1)[-1]),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=pfx,
            ),
            prefix=f"{prefix}.model.layers" if prefix else "model.layers",
        )

        # Final norm
        self.norm = RMSNorm(
            talker_cfg.hidden_size,
            eps=getattr(talker_cfg, "rms_norm_eps", 1e-6),
        )

        # Codec head: projects hidden -> codec_0 logits
        self.codec_head = ParallelLMHead(
            talker_cfg.vocab_size,
            talker_cfg.hidden_size,
            quant_config=quant_config,
            prefix="codec_head",
        )
        self.logits_processor = LogitsProcessor(talker_cfg.vocab_size)

        # Per-sequence state store
        self._state_store = get_default_sequence_store()

        # Sub-talker reference (set after weight loading via set_subtalker)
        self._subtalker_runner = None

        # Text projection MLP (loaded from weights)
        # This is Qwen3TTSTalkerResizeMLP: fc1 + act + fc2
        text_hidden = talker_cfg.text_hidden_size
        self.text_projection_fc1 = nn.Linear(text_hidden, text_hidden, bias=True)
        self.text_projection_fc2 = nn.Linear(text_hidden, talker_cfg.hidden_size, bias=True)
        self._text_proj_act = nn.SiLU()

    def set_subtalker(self, runner: Any) -> None:
        """Set the CUDA graph sub-talker runner."""
        self._subtalker_runner = runner

    def set_state_store(self, store: SequenceStateStore) -> None:
        self._state_store = store

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """vLLM model forward.

        - Prefill: ``inputs_embeds`` is provided by the preprocessor.
          ``input_ids`` is None or dummy.
        - Decode: ``input_ids`` contains sampled codec_0 tokens.
          We run the sub-talker, compute embeddings, then transformer.
        """
        if inputs_embeds is None and input_ids is not None:
            # Decode path: embed codec_0 token
            inputs_embeds = self.codec_embedding(input_ids)

        # Run transformer layers
        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.codec_head, hidden_states)
        return logits

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load weights from safetensors with talker.* prefix mapping.

        Weight key mapping:
          talker.model.layers.{i}.self_attn.q_proj -> layers.{i}.self_attn.qkv_proj (merged)
          talker.model.layers.{i}.self_attn.k_proj -> layers.{i}.self_attn.qkv_proj (merged)
          talker.model.layers.{i}.self_attn.v_proj -> layers.{i}.self_attn.qkv_proj (merged)
          talker.model.layers.{i}.self_attn.o_proj -> layers.{i}.self_attn.o_proj
          talker.model.layers.{i}.mlp.gate_proj    -> layers.{i}.mlp.gate_up_proj (merged)
          talker.model.layers.{i}.mlp.up_proj      -> layers.{i}.mlp.gate_up_proj (merged)
          talker.model.layers.{i}.mlp.down_proj    -> layers.{i}.mlp.down_proj
          talker.model.layers.{i}.*layernorm*       -> layers.{i}.*layernorm*
          talker.model.codec_embedding              -> codec_embedding
          talker.model.text_embedding               -> text_embedding
          talker.codec_head                         -> codec_head
          talker.text_projection.*                  -> text_projection_*
          talker.model.norm                         -> norm
        """
        from vllm.model_executor.layers.linear import (
            QKVParallelLinear,
            RowParallelLinear,
            ColumnParallelLinear,
        )
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        # Map merged QKV and gate_up projections
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        loaded = set()
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Only process talker weights
            if not name.startswith("talker."):
                continue

            # Strip the "talker." prefix
            name = name[len("talker."):]

            # Map HF weight names to vLLM parameter names
            mapped_name = self._map_weight_name(name)
            if mapped_name is None:
                # Sub-talker weights (code_predictor.*) — skip, loaded separately
                continue

            # Check for stacked (merged) parameters
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in mapped_name:
                    target_name = mapped_name.replace(weight_name, param_name)
                    if target_name in params_dict:
                        param = params_dict[target_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, shard_id)
                        loaded.add(f"talker.{name}")
                        is_stacked = True
                    break

            if is_stacked:
                continue

            # Direct parameter mapping
            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(f"talker.{name}")
            else:
                logger.debug("Skipping unmapped talker weight: %s -> %s", name, mapped_name)

        return loaded

    def _map_weight_name(self, hf_name: str) -> str | None:
        """Map HF weight name (without talker. prefix) to vLLM param name."""
        # Skip sub-talker weights
        if hf_name.startswith("code_predictor."):
            return None

        # model.layers.{i}.self_attn.{q,k,v,o}_proj -> layers.{i}.self_attn.*
        if hf_name.startswith("model.layers."):
            return hf_name[len("model."):]  # strip "model." -> layers.{i}...

        # model.codec_embedding -> codec_embedding
        if hf_name == "model.codec_embedding.weight":
            return "codec_embedding.weight"

        # model.text_embedding -> text_embedding
        if hf_name == "model.text_embedding.weight":
            return "text_embedding.weight"

        # model.norm -> norm
        if hf_name.startswith("model.norm."):
            return hf_name[len("model."):]  # norm.*

        # codec_head -> codec_head
        if hf_name.startswith("codec_head."):
            return hf_name

        # text_projection.linear_fc1 -> text_projection_fc1
        if hf_name.startswith("text_projection.linear_fc1."):
            return hf_name.replace("text_projection.linear_fc1.", "text_projection_fc1.")

        if hf_name.startswith("text_projection.linear_fc2."):
            return hf_name.replace("text_projection.linear_fc2.", "text_projection_fc2.")

        # Rotary embedding buffers — skip (vLLM computes its own)
        if "rotary_emb" in hf_name:
            return None

        logger.debug("Unknown talker weight: %s", hf_name)
        return None

    # ------------------------------------------------------------------
    # Helpers for preprocessor integration
    # ------------------------------------------------------------------

    def text_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the text projection MLP (fc1 -> silu -> fc2)."""
        return self.text_projection_fc2(self._text_proj_act(self.text_projection_fc1(x)))

    def prepare_prefill(self, seq_id: str, payload: PrefillPayload) -> None:
        """Register sequence state for a new generation request."""
        self._state_store.create(
            seq_id=seq_id,
            trailing_text_hidden=payload.trailing_text_hidden,
            tts_pad_embed=payload.tts_pad_embed,
            past_hidden=None,
        )

    def forward_decode_step(
        self,
        seq_id: str,
        codec_0_token: int,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """Run sub-talker and compute next inputs_embeds for the main talker.

        Called during decode to produce the full 16-codebook frame and the
        next input embedding.

        Args:
            seq_id: Sequence identifier
            codec_0_token: The sampled coarse token (codebook 0)
            hidden_state: [1, D] last hidden state from the talker

        Returns:
            (next_inputs_embeds [1, 1, D], codec_frame [16 ints])
        """
        state = self._state_store.get(seq_id)
        if state is None:
            raise KeyError(f"No sequence state for {seq_id}")

        device = hidden_state.device
        dtype = hidden_state.dtype

        # Get codec_0 embedding
        codec_0_ids = torch.tensor([[codec_0_token]], device=device, dtype=torch.long)
        codec_0_embed = self.codec_embedding(codec_0_ids)  # [1, 1, D]

        # Run sub-talker: inputs = [past_hidden, codec_0_embed] -> [1, 2, D]
        past_hidden = hidden_state.unsqueeze(0) if hidden_state.dim() == 2 else hidden_state
        sub_input = torch.cat([past_hidden, codec_0_embed], dim=1)  # [1, 2, D]

        if self._subtalker_runner is None:
            raise RuntimeError("Qwen3TTSForVLLM requires a CUDA graph subtalker runner.")

        fine_tokens = self._subtalker_runner.generate(sub_input)  # [1, 15]

        # Build full codec frame
        codec_frame = [codec_0_token] + fine_tokens[0].tolist()
        self._state_store.append_codec_frame(seq_id, codec_frame)

        # Compute next input embedding: sum of all 16 codec embeddings
        # codec_0_embed is already computed above
        # For codebooks 1-15, use sub-talker's codec_embeddings
        codec_hiddens = [codec_0_embed]  # [1, 1, D]
        for i in range(15):
            tok = fine_tokens[:, i:i+1]  # [1, 1]
            emb = self._subtalker_runner._codec_embeddings[i](tok)
            codec_hiddens.append(emb)

        # Sum all codec embeddings -> [1, 1, D]
        # But codec_hiddens have different seq dims. Let's be more careful.
        # codec_0_embed: [1, 1, 2048], each fine emb: [1, 1, 2048]
        all_embeds = torch.cat(codec_hiddens, dim=1)  # [1, 16, 2048]
        inputs_embeds = all_embeds.sum(1, keepdim=True)  # [1, 1, 2048]

        # Add text conditioning
        gen_step = state.generation_step
        trailing = state.trailing_text_hidden
        if trailing is not None and gen_step < trailing.shape[1]:
            inputs_embeds = inputs_embeds + trailing[:, gen_step:gen_step+1]
        elif state.tts_pad_embed is not None:
            inputs_embeds = inputs_embeds + state.tts_pad_embed

        # Update state
        self._state_store.advance(seq_id, past_hidden=hidden_state)

        return inputs_embeds, codec_frame

    def finalize(self, seq_id: str) -> Any:
        """Clean up sequence state."""
        return self._state_store.pop(seq_id)
