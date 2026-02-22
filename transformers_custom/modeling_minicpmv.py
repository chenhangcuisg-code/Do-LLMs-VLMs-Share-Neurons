import math
import json
import torch
from threading import Thread
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from transformers import LlamaPreTrainedModel, LlamaForCausalLM, TextIteratorStreamer
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer
from transformers import AutoProcessor
from typing import Optional, List, Dict, Tuple, Any  # === NEURON: new imports ===

from .configuration_minicpm import MiniCPMVConfig
from .resampler import Resampler

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_MEAN
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD


class MiniCPMVPreTrainedModel(LlamaPreTrainedModel):
    config_class = MiniCPMVConfig


class MiniCPMV(MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        from .modeling_llama import LlamaForCausalLMDetect as CustomLlamaModel

        print(config)
        self.text_model = CustomLlamaModel(config)
        self.llm = self.text_model

        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.transform = self.init_transform()

    def set_text_model(self, text_model):
        """
        Replace the underlying text model with a custom one that exposes neuron statistics.

        The custom ``text_model`` is expected to support ``early_exit_layers`` in ``forward``
        and return:
        ``(outputs, hidden_scores_fwd_up, hidden_scores_fwd_down, hidden_scores_q, hidden_scores_k, hidden_scores_v, hidden_scores_o)``.
        """
        self.text_model = text_model

    def init_vision_module(self):
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
        model = Idefics2VisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def init_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder
        self.text_model = decoder

    def get_decoder(self):
        return self.llm

    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                if self.config.batch_vision_input:
                    max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                    all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                       padding_value=0.0)
                    B, L, _ = all_pixel_values.shape
                    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                    patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                    for i in range(B):
                        patch_attn_mask[i, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                    vision_embedding = self.vpm(all_pixel_values.type(dtype),
                                                patch_attention_mask=patch_attn_mask).last_hidden_state
                    vision_embedding = self.resampler(vision_embedding, tgt_sizes)
                else:
                    # get vision_embedding foreach
                    vision_embedding = []
                    for single_tgt_size, single_pixel_values in zip(tgt_sizes, all_pixel_values):
                        single_pixel_values = single_pixel_values.unsqueeze(0)
                        B, L, _ = single_pixel_values.shape
                        single_pixel_values = single_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
                        single_vision_embedding = self.vpm(single_pixel_values.type(dtype)).last_hidden_state
                        single_vision_embedding = self.resampler(single_vision_embedding, single_tgt_size.unsqueeze(0))
                        vision_embedding.append(single_vision_embedding)
                    vision_embedding = torch.vstack(vision_embedding)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224),
                        device=device, dtype=dtype
                    )
                    tgt_sizes = torch.Tensor(
                        [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']

        if hasattr(self.llm.config, 'scale_emb'):
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(
            i, torch.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data['image_bound'][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                                          cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def forward(
        self,
        data: Dict[str, Any],
        *,
        early_exit_layers: Optional[List[int]] = None,
        return_hidden_scores: bool = False,
        **kwargs
    ):
        """
        Forward pass with optional neuron-statistics branch.

        If ``early_exit_layers`` or ``return_hidden_scores=True`` is requested, this method delegates
        to ``self.text_model`` which is expected to return:
        ``(outputs, fwd_up, fwd_down, q, k, v, o)``.

        Otherwise it behaves like the original MiniCPM-V forward and calls ``self.llm(...)`` directly.
        """
        vllm_embedding, _ = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        if return_hidden_scores or (early_exit_layers is not None):
            # For neuron statistics, use the extended text_model forward branch.
            # Prefer ``attention_mask`` from ``data``; fall back to ``kwargs``.
            attention_mask = data.get("attention_mask", None)
            if attention_mask is None:
                attention_mask = kwargs.get("attention_mask", None)

            outputs = self.text_model(
                input_ids=None,
                inputs_embeds=vllm_embedding,
                attention_mask=attention_mask,
                position_ids=position_ids,
                early_exit_layers=early_exit_layers,
                use_cache=kwargs.get("use_cache", None),
                output_attentions=kwargs.get("output_attentions", None),
                output_hidden_states=kwargs.get("output_hidden_states", None),
                return_dict=True,
                cache_position=kwargs.get("cache_position", None),
                activate_keys_fwd_up_set=kwargs.get("activate_keys_fwd_up_set", None),
                activate_keys_fwd_down_set=kwargs.get("activate_keys_fwd_down_set", None),
                activate_keys_q_set=kwargs.get("activate_keys_q_set", None),
                activate_keys_k_set=kwargs.get("activate_keys_k_set", None),
                activate_keys_v_set=kwargs.get("activate_keys_v_set", None),
                under_layer=kwargs.get("under_layer", None),
                gen_layer=kwargs.get("gen_layer", None),
                atten_number=kwargs.get("atten_number", None),
                ffn_number=kwargs.get("ffn_number", None),
                whether_under=kwargs.get("whether_under", None),
                whether_reason=kwargs.get("whether_reason", None),
                whether_gen=kwargs.get("whether_gen", None),
                whether_under_fwd=kwargs.get("whether_under_fwd", None),
                whether_reason_fwd=kwargs.get("whether_reason_fwd", None),
                whether_gen_fwd=kwargs.get("whether_gen_fwd", None),
            )
            if isinstance(outputs, tuple) and len(outputs) == 7:
                return outputs  # (outputs, fwd_up, fwd_down, q, k, v, o)
            else:
                # Backward-compatible: if external ``text_model`` returns only ``outputs``,
                # pad with empty dicts.
                empty = {}, {}, {}, {}, {}, {}
                return (outputs,) + empty

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )

    def _decode_text(self, result_ids, tokenizer):
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] == tokenizer.eos_id or result[-1] == tokenizer.eot_id:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def _decode(self, inputs_embeds, tokenizer, decode_text=False, **kwargs):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            **kwargs
        )
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    # === NEURON: generate 扩展，支持在生成前先跑一遍 text_model 拿 neuron 统计，并随结果一起返回 ===
    def generate(
        self,
        model_inputs,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        early_exit_layers: Optional[List[int]] = None,   # 新增
        return_hidden_scores: bool = False,               # 新增
        **kwargs
    ):
        """
        返回：
          - 当未请求 hidden scores：与原来一致，返回生成结果 (ids 或文本 / 或 streamer)。
          - 当请求 hidden scores：
              * stream=False -> 返回 (result, hidden_scores_dict)
              * stream=True  -> 返回 (streamer, hidden_scores_dict)
        hidden_scores_dict keys:
            fwd_up, fwd_down, q, k, v, o
        """
        bs = len(model_inputs["input_ids"])
        img_list = model_inputs["pixel_values"]
        tgt_sizes = model_inputs["tgt_sizes"]
        if img_list is None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)
        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(img.to(self.device))
                if img_inps:
                    pixel_values.append(img_inps)
                else:
                    pixel_values.append([])
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        (
            input_embeds,
            vision_hidden_states,
        ) = self.get_vllm_embedding(model_inputs)

        # === NEURON: 若需要统计，先跑一次 forward（text_model 分支），拿到 hidden scores ===
        hidden_scores = None
        if return_hidden_scores or (early_exit_layers is not None):
            outputs_and_scores = self.forward(
                {
                    "input_ids": model_inputs["input_ids"],
                    "position_ids": model_inputs["position_ids"],
                    "attention_mask": model_inputs.get("attention_mask", None),
                    "image_bound": model_inputs.get("image_bound", [[]] * bs),
                    "pixel_values": model_inputs.get("pixel_values", [[]] * bs),
                    "tgt_sizes": model_inputs.get("tgt_sizes", None),
                },
                early_exit_layers=early_exit_layers,
                return_hidden_scores=True,
                use_cache=kwargs.get("use_cache", None),
                output_attentions=kwargs.get("output_attentions", None),
                output_hidden_states=kwargs.get("output_hidden_states", None),
                cache_position=kwargs.get("cache_position", None),
                activate_keys_fwd_up_set=kwargs.get("activate_keys_fwd_up_set", None),
                activate_keys_fwd_down_set=kwargs.get("activate_keys_fwd_down_set", None),
                activate_keys_q_set=kwargs.get("activate_keys_q_set", None),
                activate_keys_k_set=kwargs.get("activate_keys_k_set", None),
                activate_keys_v_set=kwargs.get("activate_keys_v_set", None),
                under_layer=kwargs.get("under_layer", None),
                gen_layer=kwargs.get("gen_layer", None),
                atten_number=kwargs.get("atten_number", None),
                ffn_number=kwargs.get("ffn_number", None),
                whether_under=kwargs.get("whether_under", None),
                whether_reason=kwargs.get("whether_reason", None),
                whether_gen=kwargs.get("whether_gen", None),
                whether_under_fwd=kwargs.get("whether_under_fwd", None),
                whether_reason_fwd=kwargs.get("whether_reason_fwd", None),
                whether_gen_fwd=kwargs.get("whether_gen_fwd", None),
            )
            # 解析七元组
            if isinstance(outputs_and_scores, tuple) and len(outputs_and_scores) == 7:
                _outputs, fwd_up, fwd_down, q, k, v, o = outputs_and_scores
                hidden_scores = {
                    "fwd_up": fwd_up,
                    "fwd_down": fwd_down,
                    "q": q,
                    "k": k,
                    "v": v,
                    "o": o,
                }
            else:
                hidden_scores = {"fwd_up": {}, "fwd_down": {}, "q": {}, "k": {}, "v": {}, "o": {}}

        # === 正常 decode / 或流式 ===
        if stream:
            kwargs.pop("decode_text", None)
            res = self._decode_stream(input_embeds, tokenizer, **kwargs)
            return (res, hidden_scores) if hidden_scores is not None else res
        else:
            res = self._decode(input_embeds, tokenizer, **kwargs)
            return (res, hidden_scores) if hidden_scores is not None else res

    # === NEURON: chat 扩展，支持把 hidden scores 一并返回 ===
    def chat(
            self,
            image,
            msgs,
            tokenizer,
            processor=None,
            vision_hidden_states=None,
            max_new_tokens=1024,
            sampling=True,
            max_inp_length=2048,
            system_prompt='',
            stream=False,
            # 新增：neuron 相关参数（直传到 generate / forward）
            early_exit_layers: Optional[List[int]] = None,
            return_hidden_scores: bool = False,
            **kwargs
    ):
        if processor is None:
            processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        copy_msgs = deepcopy(msgs)

        assert len(msgs) > 0, "msgs is empty"
        assert sampling or not stream, "if use stream mode, make sure sampling=True"

        if image is not None and isinstance(copy_msgs[0]["content"], str):
            # copy_msgs[0]['content'] = '(<image>./</image>)\n' + copy_msgs[0]['content']
            copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        if system_prompt:
            sys_msg = {'role': 'system', 'content': system_prompt}
            copy_msgs = [sys_msg] + copy_msgs

        prompt = processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images, return_tensors="pt", max_length=max_inp_length).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )
        with torch.inference_mode():
            res = self.generate(
                inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                decode_text=True,
                early_exit_layers=early_exit_layers,         # === 透传 ===
                return_hidden_scores=return_hidden_scores,   # === 透传 ===
                **generation_config
            )

        # 根据是否请求 hidden scores、是否流式，组织返回
        if stream:
            # res 可能是 streamer 或 (streamer, hidden_scores)
            if isinstance(res, tuple):
                streamer, hidden_scores = res

                def stream_gen():
                    for text in streamer:
                        text = text.replace(tokenizer.eot_token, '').replace(tokenizer.eos_token, '')
                        yield text

                # 返回 (生成器, hidden_scores)
                return stream_gen(), hidden_scores
            else:
                def stream_gen():
                    for text in res:
                        text = text.replace(tokenizer.eot_token, '').replace(tokenizer.eos_token, '')
                        yield text

                return stream_gen()
        else:
            # res 可能是 文本 或 (文本, hidden_scores)
            if isinstance(res, tuple):
                answer, hidden_scores = res
                return answer[0], hidden_scores  # decode_text=True -> list[str]，取第一个 batch
            else:
                answer = res[0]
                return answer
