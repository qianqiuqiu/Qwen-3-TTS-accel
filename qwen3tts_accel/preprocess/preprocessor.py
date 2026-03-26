"""Preprocessor: builds talker inputs from text + voice clone prompt.

Uses the HF model's embedding and projection layers to construct the exact
same ``inputs_embeds``, ``trailing_text_hidden``, and ``tts_pad_embed``
that ``Qwen3TTSForConditionalGeneration.generate()`` would build internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PrefillPayload:
    request_id: str
    text: str
    language: str
    engine_prompt: str
    inputs_embeds: torch.Tensor          # [1, L, D]
    attention_mask: torch.Tensor         # [1, L]
    trailing_text_hidden: torch.Tensor   # [1, T, D]
    tts_pad_embed: torch.Tensor          # [1, 1, D]
    voice_clone_prompt: Any              # original dict (for codec decode)


class Qwen3TTSPreprocessor:
    """Converts (text, language, voice_clone_prompt) → PrefillPayload."""

    def __init__(self, model: Any, tokenizer: Any | None = None) -> None:
        # model can be either:
        #   - Qwen3TTSModel (inference wrapper): has .model and .processor
        #   - Qwen3TTSForConditionalGeneration (HF model): has .talker directly
        if hasattr(model, "model") and hasattr(model, "processor"):
            # Inference wrapper
            self._wrapper = model
            self._hf_model = model.model        # Qwen3TTSForConditionalGeneration
            self._tokenizer = model.processor    # Qwen3TTSProcessor
        else:
            self._wrapper = None
            self._hf_model = model
            self._tokenizer = tokenizer

        self._talker = self._hf_model.talker
        self._config = self._hf_model.config
        self._talker_config = self._hf_model.config.talker_config
        self._device = next(self._talker.parameters()).device
        self._dtype = next(self._talker.parameters()).dtype

    @torch.no_grad()
    def prepare_inputs(
        self,
        *,
        request_id: str,
        text: str,
        language: str,
        voice_clone_prompt: Any,
    ) -> PrefillPayload:
        # --- tokenize ---
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        tok_out = self._tokenizer(assistant_text, return_tensors="pt", padding=True)
        input_ids = tok_out["input_ids"].to(self._device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # --- voice clone prompt dict ---
        vcp_dict = self._to_prompt_dict(voice_clone_prompt)

        # --- ref_ids for ICL ---
        ref_ids = None
        if vcp_dict is not None and vcp_dict.get("icl_mode", [False])[0]:
            # find ref_text from the original prompt items
            ref_text = self._get_ref_text(voice_clone_prompt)
            if ref_text:
                ref_str = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
                ref_tok = self._tokenizer(ref_str, return_tensors="pt", padding=True)
                ref_ids_t = ref_tok["input_ids"].to(self._device)
                if ref_ids_t.dim() == 1:
                    ref_ids_t = ref_ids_t.unsqueeze(0)
                ref_ids = [ref_ids_t]

        # --- delegate to HF model.generate() for correct embed construction ---
        # We call generate with max_new_tokens=0 to get inputs_embeds only.
        # Actually, we replicate the exact embedding logic from generate().

        # Determine speaker and language
        spk_embed = None
        if vcp_dict is not None:
            if vcp_dict.get("x_vector_only_mode", [False])[0] or vcp_dict.get("icl_mode", [False])[0]:
                spk_embed = vcp_dict["ref_spk_embedding"][0].to(self._device).to(self._dtype)

        lang_lower = language.lower() if language else "auto"
        if lang_lower == "auto" or lang_lower not in self._talker_config.codec_language_id:
            language_id = None
        else:
            language_id = self._talker_config.codec_language_id[lang_lower]

        # --- compute special embeddings ---
        special_ids = torch.tensor(
            [[self._config.tts_bos_token_id,
              self._config.tts_eos_token_id,
              self._config.tts_pad_token_id]],
            device=self._device,
            dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._talker.text_projection(
            self._talker.get_text_embeddings()(special_ids)
        ).chunk(3, dim=1)  # each [1, 1, D]

        # --- codec prefill tokens ---
        if language_id is None:
            codec_prefill_ids = [[
                self._talker_config.codec_nothink_id,
                self._talker_config.codec_think_bos_id,
                self._talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill_ids = [[
                self._talker_config.codec_think_id,
                self._talker_config.codec_think_bos_id,
                language_id,
                self._talker_config.codec_think_eos_id,
            ]]

        codec_embed_0 = self._talker.get_input_embeddings()(
            torch.tensor(codec_prefill_ids, device=self._device, dtype=input_ids.dtype)
        )
        codec_embed_1 = self._talker.get_input_embeddings()(
            torch.tensor(
                [[self._talker_config.codec_pad_id, self._talker_config.codec_bos_id]],
                device=self._device, dtype=input_ids.dtype,
            )
        )

        if spk_embed is not None:
            codec_input_embedding = torch.cat(
                [codec_embed_0, spk_embed.view(1, 1, -1), codec_embed_1], dim=1
            )
        else:
            codec_input_embedding = torch.cat([codec_embed_0, codec_embed_1], dim=1)

        # --- role embed: <|im_start|>assistant\n ---
        role_embed = self._talker.text_projection(
            self._talker.get_text_embeddings()(input_ids[:, :3])
        )

        # --- tts_pad * N + tts_bos overlay on codec (minus last token) ---
        n_codec_prefix = codec_input_embedding.shape[1] - 2  # exclude last 2 for pad structure
        talker_prefix = torch.cat(
            (tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), tts_bos_embed),
            dim=1,
        ) + codec_input_embedding[:, :-1]

        talker_input_embed = torch.cat((role_embed, talker_prefix), dim=1)

        # --- ICL or streaming text ---
        if (vcp_dict is not None
                and vcp_dict.get("ref_code") is not None
                and vcp_dict.get("icl_mode", [False])[0]
                and ref_ids is not None):
            # ICL mode: use generate_icl_prompt logic
            ref_code = vcp_dict["ref_code"][0].to(self._device)
            icl_embed, trailing_text_hidden = self._hf_model.generate_icl_prompt(
                text_id=input_ids[:, 3:-5],
                ref_id=ref_ids[0][:, 3:-2],
                ref_code=ref_code,
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=False,
            )
            talker_input_embed = torch.cat([talker_input_embed, icl_embed], dim=1)
        else:
            # Non-ICL streaming: first text token + remaining as trailing
            first_text_embed = self._talker.text_projection(
                self._talker.get_text_embeddings()(input_ids[:, 3:4])
            ) + codec_input_embedding[:, -1:]
            talker_input_embed = torch.cat([talker_input_embed, first_text_embed], dim=1)

            # trailing_text_hidden = text_projection(remaining text) + eos
            trailing_text_hidden = torch.cat((
                self._talker.text_projection(
                    self._talker.get_text_embeddings()(input_ids[:, 4:-5])
                ),
                tts_eos_embed,
            ), dim=1)

        # --- attention mask (all ones, no padding for single request) ---
        seq_len = talker_input_embed.shape[1]
        attention_mask = torch.ones(1, seq_len, device=self._device, dtype=torch.long)

        return PrefillPayload(
            request_id=request_id,
            text=text,
            language=language,
            engine_prompt=assistant_text,
            inputs_embeds=talker_input_embed,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            voice_clone_prompt=vcp_dict,
        )

    def _to_prompt_dict(self, voice_clone_prompt: Any) -> dict | None:
        if voice_clone_prompt is None:
            return None
        if isinstance(voice_clone_prompt, dict):
            return voice_clone_prompt
        if isinstance(voice_clone_prompt, list):
            # list of VoiceClonePromptItem
            return {
                "ref_code": [getattr(it, "ref_code", None) for it in voice_clone_prompt],
                "ref_spk_embedding": [getattr(it, "ref_spk_embedding", None) for it in voice_clone_prompt],
                "x_vector_only_mode": [getattr(it, "x_vector_only_mode", False) for it in voice_clone_prompt],
                "icl_mode": [getattr(it, "icl_mode", False) for it in voice_clone_prompt],
            }
        return None

    def _get_ref_text(self, voice_clone_prompt: Any) -> str | None:
        if isinstance(voice_clone_prompt, list) and voice_clone_prompt:
            return getattr(voice_clone_prompt[0], "ref_text", None)
        if isinstance(voice_clone_prompt, dict):
            rt = voice_clone_prompt.get("ref_text")
            if isinstance(rt, list) and rt:
                return rt[0]
            return rt
        return None
