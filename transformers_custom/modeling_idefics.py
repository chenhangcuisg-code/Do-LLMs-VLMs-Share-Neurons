# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Idefics3 model (extended with neuron selection / early-exit plumbing)."""

from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Tuple, Dict

import itertools
import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging
from transformers.models.auto import AutoModel
from transformers.models.idefics3.configuration_idefics3 import Idefics3Config, Idefics3VisionConfig
from .generation import GenerationMixinCustom

logger = logging.get_logger(__name__)
from transformers import AutoConfig, AutoModel
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from .modeling_llama import LlamaModel as CustomLlamaModel

# Register your custom model
# MODEL_MAPPING.register(
#     AutoConfig.for_model("llama"),
#     CustomLlamaModel
# )

@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Idefics3 model's outputs that may also contain a past key/values (to speed up sequential decoding).
    """
)
class Idefics3BaseModelOutputWithPast(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the model.
        If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
        hidden_size)` is output.
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
        `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
        encoder_sequence_length, embed_size_per_head)`.
        Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
        `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
        input) to speed up sequential decoding.
    image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
        sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Idefics causal language model (or autoregressive) outputs.
    """
)
class Idefics3CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
        sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[tuple[torch.FloatTensor]] = None


# Copied from transformers.models.idefics2.modeling_idefics2.Idefics2VisionEmbeddings with Idefics2->Idefics3
class Idefics3VisionEmbeddings(nn.Module):
    """
    This is a modified version of `siglip.modelign_siglip.SiglipVisionEmbeddings` to enable images of variable
    resolution.

    The modifications are adapted from [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://huggingface.co/papers/2307.06304)
    which allows treating images in their native aspect ratio and without the need to resize them to the same
    fixed size. In particular, we start from the original pre-trained SigLIP model
    (which uses images of fixed-size square images) and adapt it by training on images of variable resolutions.
    """

    def __init__(self, config: Idefics3VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side, device=pixel_values.device
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0, device=pixel_values.device
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            h_indices = torch.arange(nb_patches_h, device=position_ids.device, dtype=position_ids.dtype)
            w_indices = torch.arange(nb_patches_w, device=position_ids.device, dtype=position_ids.dtype)

            fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
            fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


# Copied from transformers.models.siglip.modeling_siglip.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.siglip.modeling_siglip.SiglipAttention with Siglip->Idefics3Vision
class Idefics3VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Ignore copy
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.siglip.modeling_siglip.SiglipMLP with Siglip->Idefics3Vision
class Idefics3VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Idefics3SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)


# Copied from transformers.models.idefics2.modeling_idefics2.Idefics2EncoderLayer with Idefics2->Idefics3
class Idefics3EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Idefics3VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Idefics3VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Idefics3VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Copied from transformers.models.siglip.modeling_siglip.SiglipEncoderLayer.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.siglip.modeling_siglip.SiglipEncoder with Siglip->Idefics3
class Idefics3Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Idefics3EncoderLayer`].

    Args:
        config: Idefics3Config
    """

    def __init__(self, config: Idefics3Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Idefics3EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Idefics3
class Idefics3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Idefics3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Idefics3Connector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = Idefics3SimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


@auto_docstring
class Idefics3PreTrainedModel(PreTrainedModel):
    config: Idefics3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Idefics3VisionAttention", "Idefics3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _supports_attention_backend = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, Idefics3RMSNorm):
            module.weight.data.fill_(1.0)


@auto_docstring(
    custom_intro="""
    The Idefics3 Vision Transformer Model outputting raw image embedding.
    """
)
class Idefics3VisionTransformer(Idefics3PreTrainedModel):
    config: Idefics3VisionConfig
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True

    def __init__(self, config: Idefics3VisionConfig):
        super().__init__(config)
        embed_dim = config.hidden_size

        self.embeddings = Idefics3VisionEmbeddings(config)
        self.encoder = Idefics3Encoder(config)
        self.patch_size = config.patch_size
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2VisionTransformer.get_input_embeddings
    def get_input_embeddings(self):
        return self.embeddings

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2VisionTransformer.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not self._use_flash_attention_2:
            patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
        elif not torch.any(~patch_attention_mask):
            patch_attention_mask = None

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Idefics3 model consisting of a SIGLIP vision encoder and Llama3 language decoder
    """
)
class Idefics3Model(Idefics3PreTrainedModel):
    def __init__(self, config: Idefics3Config):
        super().__init__(config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model = Idefics3VisionTransformer._from_config(config.vision_config)
        self.connector = Idefics3Connector(config)
        from .modeling_llama import LlamaModel as CustomLlamaModel

        # If only initializing
        self.text_model = CustomLlamaModel(config.text_config).to(torch.float16)
        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )
        self.image_token_id = self.config.image_token_id

        self._use_flash_attention_2 = config.text_config._attn_implementation == "flash_attention_2"

        self.post_init()

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2Model.enable_input_require_grads
    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.

        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032

        Override to set output.requires_grad = True for both the decoder's and vision model's embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = get_lowest_module(self.vision_model).register_forward_hook(
            make_inputs_require_grads
        )

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2Model.disable_input_require_grads
    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2Model.get_input_embeddings
    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2Model.set_input_embeddings
    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        """
        Merge token embeddings with image hidden states by replacing <image_token_id> slots in the text stream.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_hidden_states = image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_hidden_states)
        return inputs_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, pixel_attention_mask: torch.LongTensor = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            pixel_attention_mask (`torch.LongTensor`, *optional*):
                The attention mask indicating padded regions in the image.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
        image_hidden_states.last_hidden_state

        # Modality projection & resampling
        image_hidden_states = self.connector(image_hidden_states.last_hidden_state)
        return image_hidden_states

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        # ===== Extended neuron/early-exit controls (pass-through to text_model) =====
        early_exit_layers: Optional[List[int]] = None,
        activate_keys_fwd_up_set: Optional[Dict[int, List[int]]] = None,
        activate_keys_fwd_down_set: Optional[Dict[int, List[int]]] = None,
        activate_keys_q_set: Optional[Dict[int, List[int]]] = None,
        activate_keys_k_set: Optional[Dict[int, List[int]]] = None,
        activate_keys_v_set: Optional[Dict[int, List[int]]] = None,
        under_layer: Optional[int] = None,
        gen_layer: Optional[int] = None,
        atten_number: Optional[int] = None,
        ffn_number: Optional[int] = None,
        whether_under: Optional[bool] = None,
        whether_reason: Optional[bool] = None,
        whether_gen: Optional[bool] = None,
        whether_under_fwd: Optional[bool] = None,
        whether_reason_fwd: Optional[bool] = None,
        whether_gen_fwd: Optional[bool] = None,
        # ========================================================================
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, Idefics3BaseModelOutputWithPast]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )
        # print('@@@@@@@@@@@@')
        # print(type(self.text_model))
        # # 或者
        # print(self.text_model.__class__)
        # print(early_exit_layers)
        # print('@@@@@@@@@@@@')
        # ====== forward text_model (supports extended returns if early_exit is enabled in text_model) ======
        text_outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            # pass-through neuron/early-exit plumbing (your text_model is already modified to accept these)
            early_exit_layers=early_exit_layers,
            activate_keys_fwd_up_set=activate_keys_fwd_up_set,
            activate_keys_fwd_down_set=activate_keys_fwd_down_set,
            activate_keys_q_set=activate_keys_q_set,
            activate_keys_k_set=activate_keys_k_set,
            activate_keys_v_set=activate_keys_v_set,
            under_layer=under_layer,
            gen_layer=gen_layer,
            atten_number=atten_number,
            ffn_number=ffn_number,
            whether_under=whether_under,
            whether_reason=whether_reason,
            whether_gen=whether_gen,
            whether_under_fwd=whether_under_fwd,
            whether_reason_fwd=whether_reason_fwd,
            whether_gen_fwd=whether_gen_fwd,
            **kwargs,
        )
        # print('##################TEXT OUTPUTs #########################')
        # print(text_outputs)

        # text_outputs can be either a BaseModelOutputWithPast-like object
        # or a 7-tuple: (outputs, fwd_up, fwd_down, q, k, v, o)
        extra_scores = None
        if isinstance(text_outputs, tuple) and len(text_outputs) == 7 and isinstance(text_outputs[1], dict):
            outputs, hidden_scores_fwd_up, hidden_scores_fwd_down, hidden_scores_q, hidden_scores_k, hidden_scores_v, hidden_scores_o = text_outputs
            extra_scores = (hidden_scores_fwd_up, hidden_scores_fwd_down, hidden_scores_q, hidden_scores_k, hidden_scores_v, hidden_scores_o)
        else:
            outputs = text_outputs

        final = Idefics3BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )

        if extra_scores is not None:
            return (final, *extra_scores)
        return final


@auto_docstring(
    custom_intro="""
    The Idefics3 Model with a language modeling head. It is made up a SigLIP vision encoder, with a language modeling head on top.
    """
)
class Idefics3ForConditionalGeneration(Idefics3PreTrainedModel, GenerationMixinCustom):
    _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2ForConditionalGeneration.__init__ with Idefics2->Idefics3
    def __init__(self, config):
        super().__init__(config)
        self.model = Idefics3Model(config)
        self.image_token_id = self.config.image_token_id

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2ForConditionalGeneration.enable_input_require_grads
    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = self.model.vision_model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grads
        )

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2ForConditionalGeneration.disable_input_require_grads
    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2ForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.model.text_model.get_input_embeddings()

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2ForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        self.model.text_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.FloatTensor, pixel_attention_mask: torch.LongTensor = None):
        return self.model.get_image_features(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,  # default True to collect layer states when probing
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # ======== neuron selection / early exit controls ========
        early_exit_layers: Optional[List[int]] = None,
        top_ratio_atten: float = 0.0,
        top_ratio_ffn: float = 0.0,
        activate_keys_fwd_up_set: Optional[Dict[int, List[int]]] = None,
        activate_keys_fwd_down_set: Optional[Dict[int, List[int]]] = None,
        activate_keys_q_set: Optional[Dict[int, List[int]]] = None,
        activate_keys_k_set: Optional[Dict[int, List[int]]] = None,
        activate_keys_v_set: Optional[Dict[int, List[int]]] = None,
        under_layer: Optional[int] = None,
        gen_layer: Optional[int] = None,
        atten_number: Optional[int] = None,
        ffn_number: Optional[int] = None,
        whether_under: Optional[bool] = None,
        whether_reason: Optional[bool] = None,
        whether_gen: Optional[bool] = None,
        whether_under_fwd: Optional[bool] = None,
        whether_reason_fwd: Optional[bool] = None,
        whether_gen_fwd: Optional[bool] = None,
        # =======================================================
        **kwargs,
    ) -> Union[tuple, Idefics3CausalLMOutputWithPast]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `Idefics3ForConditionalGeneration`).
            Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
            computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # forward backbone
        model_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            # pass-through neuron/early-exit controls
            early_exit_layers=early_exit_layers,
            activate_keys_fwd_up_set=activate_keys_fwd_up_set,
            activate_keys_fwd_down_set=activate_keys_fwd_down_set,
            activate_keys_q_set=activate_keys_q_set,
            activate_keys_k_set=activate_keys_k_set,
            activate_keys_v_set=activate_keys_v_set,
            under_layer=under_layer,
            gen_layer=gen_layer,
            atten_number=atten_number,
            ffn_number=ffn_number,
            whether_under=whether_under,
            whether_reason=whether_reason,
            whether_gen=whether_gen,
            whether_under_fwd=whether_under_fwd,
            whether_reason_fwd=whether_reason_fwd,
            whether_gen_fwd=whether_gen_fwd,
            **kwargs,
        )
        # print('MODEL OUT#######')
        # print(model_out)
        # print(isinstance(model_out, tuple))
        # print(model_out)
        # Unpack if neuron-probing tuple is returned
        if isinstance(model_out, tuple) and len(model_out) == 7 and isinstance(model_out[1], dict):
            outputs, hidden_scores_fwd_up, hidden_scores_fwd_down, hidden_scores_q, hidden_scores_k, hidden_scores_v, hidden_scores_o = model_out
        else:
            outputs = model_out
            hidden_scores_fwd_up = hidden_scores_fwd_down = hidden_scores_q = hidden_scores_k = hidden_scores_v = hidden_scores_o = None

        hidden_states = outputs[0]  # last_hidden_state
        # compute logits (optionally slicing trailing steps)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        base_out = Idefics3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states if hasattr(outputs, "image_hidden_states") else None,
        )

        # If early-exit requested, compute logits_dict + activation index sets (mirroring your Llama example)
        # === REPLACE the early-exit block with the code below ===
        if early_exit_layers is not None and hidden_scores_fwd_up is not None:
            logits_dict = {}
            activate_keys_fwd_up = {}
            activate_keys_fwd_down = {}
            activate_keys_q = {}
            activate_keys_k = {}
            activate_keys_v = {}
            activate_keys_o = {}
            no_use_layer_index = []

            # 1) Aggregate scores per layer and build ``combined_data`` (weights 3:2:2).
            #    Use ``np.sum`` to support both lists and ndarrays.
            summed_data_fwd = {k: float(np.sum(v)) for k, v in hidden_scores_fwd_up.items()}
            summed_data_q = {k: float(np.sum(v)) for k, v in hidden_scores_q.items()}
            summed_data_v = {k: float(np.sum(v)) for k, v in hidden_scores_v.items()}

            combined_data = {
                k: (summed_data_fwd[k] * 3.0 + summed_data_q[k] * 2.0 + summed_data_v[k] * 2.0)
                for k in summed_data_fwd
            }

            def _topn(arr, n: int):
                """Return the indices of the top-n values in descending order; empty if ``n <= 0``."""
                if n <= 0:
                    return np.empty((0,), dtype=int)
                a = np.asarray(arr)
                n = min(n, a.shape[0])
                # ``np.argsort`` is ascending; take last ``n`` and then reverse to get descending order.
                return np.argsort(a)[-n:][::-1]

            def _select_by_ratio(arr, ratio: float = 0.6):
                a = np.asarray(arr)
                if a.size == 0:
                    return np.empty((0,), dtype=int)
                thr = float(a.max()) * ratio
                return np.where(a >= thr)[0]

            for early_layer in early_exit_layers:
                # 2) Per-layer logits for this layer（若未保留 hidden_states，回退到最后一层）
                if outputs.hidden_states is None or early_layer >= len(outputs.hidden_states):
                    layer_h = outputs.last_hidden_state
                else:
                    layer_h = outputs.hidden_states[early_layer]
                logits_dict[early_layer] = self.lm_head(layer_h)

                # 3) Compute Top-N counts per activation type（保持与你示例一致的来源）
                top_number_attn = int(top_ratio_atten * len(hidden_scores_fwd_up[early_layer]))
                top_number_ffn = int(top_ratio_ffn * len(hidden_scores_q[early_layer]))
                top_number_layer = 10  # Layer limit for no_use_layer_index 的层数上限

                # 4) Take Top-N indices from each dict（降序）
                activate_keys_fwd_up[early_layer] = _topn(hidden_scores_fwd_up[early_layer], top_number_ffn)
                activate_keys_fwd_down[early_layer] = _topn(hidden_scores_fwd_down[early_layer], top_number_ffn)
                activate_keys_q[early_layer] = _topn(hidden_scores_q[early_layer], top_number_attn)
                activate_keys_k[early_layer] = _topn(hidden_scores_k[early_layer], top_number_attn)
                activate_keys_v[early_layer] = _topn(hidden_scores_v[early_layer], top_number_attn)
                activate_keys_o[early_layer] = _topn(hidden_scores_o[early_layer], top_number_attn)

            # 5) Select no_use_layer_index（与示例相同：按 combined_data 升序排序后取最后 N 个）
            sorted_items = sorted(combined_data.items(), key=lambda item: item[1])
            no_use_layer_index = [item[0] for item in sorted_items[-top_number_layer:]]

            return (
                logits_dict,
                base_out,
                activate_keys_fwd_up,
                activate_keys_fwd_down,
                activate_keys_q,
                activate_keys_k,
                activate_keys_v,
                activate_keys_o,
                no_use_layer_index,
            )
        # === end of replacement ===

        # if early_exit_layers is not None and hidden_scores_fwd_up is not None:
        #     logits_dict: Dict[int, torch.Tensor] = {}
        #     activate_keys_fwd_up: Dict[int, np.ndarray] = {}
        #     activate_keys_fwd_down: Dict[int, np.ndarray] = {}
        #     activate_keys_q: Dict[int, np.ndarray] = {}
        #     activate_keys_k: Dict[int, np.ndarray] = {}
        #     activate_keys_v: Dict[int, np.ndarray] = {}
        #     activate_keys_o: Dict[int, np.ndarray] = {}
        #     no_use_layer_index: List[int] = []
        #
        #     # aggregate for layer ranking (same recipe as your sample)
        #     summed_data_fwd = {key: sum(value) for key, value in hidden_scores_fwd_up.items()}
        #     summed_data_q = {key: sum(value) for key, value in hidden_scores_q.items()}
        #     summed_data_v = {key: sum(value) for key, value in hidden_scores_v.items()}
        #     combined_data = {key: summed_data_fwd[key] * 3 + summed_data_q[key] * 2 + summed_data_v[key] * 2 for key in summed_data_fwd}
        #
        #     def top_indices_of(arr: List[float], topn: int) -> np.ndarray:
        #         if topn <= 0:
        #             return np.array([], dtype=int)
        #         a = np.asarray(arr)
        #         if topn >= a.shape[0]:
        #             return np.arange(a.shape[0], dtype=int)
        #         idx = np.argpartition(a, -topn)[-topn:]
        #         # sort descending
        #         return idx[np.argsort(-a[idx])]
        #
        #     for early_layer in early_exit_layers:
        #         # per-layer logits from that layer's hidden state
        #         if outputs.hidden_states is None or early_layer >= len(outputs.hidden_states):
        #             # fall back to last hidden if hidden_states not kept
        #             layer_h = outputs.last_hidden_state
        #         else:
        #             layer_h = outputs.hidden_states[early_layer]
        #         logits_dict[early_layer] = self.lm_head(layer_h)
        #
        #         # NOTE: follow your sample's exact "top_number" source lengths
        #         top_number_attn = int(top_ratio_atten * len(hidden_scores_fwd_up[early_layer]))
        #         top_number_ffn = int(top_ratio_ffn * len(hidden_scores_q[early_layer]))
        #         top_number_layer = 10
        #
        #         activate_keys_fwd_up[early_layer] = top_indices_of(hidden_scores_fwd_up[early_layer], top_number_ffn)
        #         activate_keys_fwd_down[early_layer] = top_indices_of(hidden_scores_fwd_down[early_layer], top_number_ffn)
        #         activate_keys_q[early_layer] = top_indices_of(hidden_scores_q[early_layer], top_number_attn)
        #         activate_keys_k[early_layer] = top_indices_of(hidden_scores_k[early_layer], top_number_attn)
        #         activate_keys_v[early_layer] = top_indices_of(hidden_scores_v[early_layer], top_number_attn)
        #         activate_keys_o[early_layer] = top_indices_of(hidden_scores_o[early_layer], top_number_attn)
        #
        #     # choose "no_use_layer_index" exactly like your sample code
        #     sorted_items = sorted(combined_data.items(), key=lambda item: item[1])
        #     no_use_layer_index = [item[0] for item in sorted_items[-top_number_layer:]]
        #
        #     return (
        #         logits_dict,
        #         base_out,
        #         activate_keys_fwd_up,
        #         activate_keys_fwd_down,
        #         activate_keys_q,
        #         activate_keys_k,
        #         activate_keys_v,
        #         activate_keys_o,
        #         no_use_layer_index,
        #     )
        #
        # # default single output path
        # return base_out

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2ForConditionalGeneration.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- there are mutually exclusive inputs (if the logic to make `image_hidden_states` take
        # precedence is moved to the model, we can remove this fn)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if image_hidden_states is not None or (cache_position is not None and cache_position[0] != 0):
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs


__all__ = ["Idefics3ForConditionalGeneration", "Idefics3PreTrainedModel", "Idefics3Model", "Idefics3VisionTransformer"]
