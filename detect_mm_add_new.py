import os, sys, json, random, argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from transformers import AutoTokenizer, AutoConfig, AutoProcessor

# === 你现有的自定义模型 ===
# from transformers_custom.modeling_gemma3 import Gemma3ForConditionalGeneration
from transformers_custom.modeling_llama import LlamaForCausalLMDetect
# from transformers_custom.modeling_gemma2 import Gemma2ForCausalLM
from transformers_custom.modeling_qwen2 import Qwen2ForCausalLM
# from transformers_custom.modeling_qwen3 import Qwen3ForCausalLM
# from transformers_custom.modeling_gemma import GemmaForCausalLM

# === NEW: 引入 MiniCPM-V / Idefics3（按你的本地命名来改动 import 路径即可） ===
# 下面这两个类名/路径以你的工程为准；如命名不同，只需改成你实际的类与模块即可。
try:
    from transformers_custom.modeling_minicpmv import MiniCPMV as MiniCPMVForCausalLM
except (ImportError, ModuleNotFoundError):
    MiniCPMVForCausalLM = None
try:
    from transformers_custom.modeling_idefics import Idefics3ForConditionalGeneration
except (ImportError, ModuleNotFoundError):
    Idefics3ForConditionalGeneration = None
try:
    from transformers_custom.modeling_llava_next import LlavaNextForConditionalGeneration
except (ImportError, ModuleNotFoundError):
    LlavaNextForConditionalGeneration = None

def _downscale_images_preserving_aspect(
    images,
    max_side: int = 1024,          # 单张图最大边
    max_megapixels_per_image: float = 1.0,  # 单张图最大像素(MP)
    max_total_megapixels: float = 2.0,      # 一次样本中多图总像素上限(MP)
    resample = Image.BICUBIC
):
    """
    等比例缩小一组 PIL 图像，限制单图最大边、单图像素上限、以及多图总像素上限。
    推荐阈值（8B, FP16）：
      - max_side=1024
      - max_megapixels_per_image=1.0（~100万像素，约1000x1000）
      - max_total_megapixels=2.0（多图合计 ~200万像素）
    """
    def _shrink_to(image: Image.Image, target_mp: float, max_side_px: int) -> Image.Image:
        w, h = image.size
        # 先按最大边限制
        scale_side = min(1.0, max_side_px / max(w, h)) if max(w, h) > max_side_px else 1.0
        # 再按 MP 限制
        cur_mp = (w * h) / 1_000_000.0
        scale_mp = (target_mp / cur_mp) ** 0.5 if cur_mp > target_mp else 1.0
        scale = min(scale_side, scale_mp)
        if scale < 1.0:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            image = image.resize((nw, nh), resample=resample)
        # 统一 RGB，去掉 alpha/EXIF 以省内存
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    # 先处理单图限制
    imgs = [ _shrink_to(im, max_megapixels_per_image, max_side) for im in images ]

    # 再看多图总像素是否超限；超限则**等比**再统一缩一遍
    total_pixels = sum(im.size[0] * im.size[1] for im in imgs)
    total_mp = total_pixels / 1_000_000.0
    if total_mp > max_total_megapixels and total_pixels > 0:
        # 统一按 sqrt 比例缩小
        scale = (max_total_megapixels / total_mp) ** 0.5
        new_imgs = []
        for im in imgs:
            w, h = im.size
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            if (nw, nh) != (w, h):
                im = im.resize((nw, nh), resample=resample)
            if im.mode != "RGB":
                im = im.convert("RGB")
            new_imgs.append(im)
        imgs = new_imgs

    return imgs

# =========================
# 数据加载（含 multimodal）
# =========================
def load_lines_from_dataset(task, lang, args):
    """
    Load dataset lines or (image(s), text) samples.
    Supports:
      - 'multimodal'   -> 读取 Parquet，需包含 problem, answer(可选), images(列表：路径或Base64)
      - 'gsm'          -> mgsm_en.tsv
    返回统一的样本结构：
      - 文本任务: str（prompt 文本）
      - 多模态任务: dict { "images": List[str|PIL|bytes], "prompt": str, "answer": Optional[str] }
    """
    if task == "multimodal":
        if not getattr(args, "multimodal_file", None):
            raise ValueError("请提供 --multimodal_file 参数，指向包含 problem, answer, images 列的 Parquet 文件。")
        df = pd.read_parquet(args.multimodal_file)

        instruction_following = (
            r'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. '
            r'The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.'
        )
        samples = []
        for _, row in df.iterrows():
            images = row['images']
            prompt = f"{row['problem']} {instruction_following}"
            samples.append({
                "images": images,
                "prompt": prompt,
                "answer": row.get('answer', None)
            })
        return samples

    elif task == "gsm":
        if lang != "en":
            raise ValueError("Only English ('en') GSM data is supported.")
        file_path = f"{args.corpus_path}/llama3ds_math.tsv"
        df = pd.read_csv(file_path, sep='\t', names=["question", "answer"])
        return [
            f"Question: {q}\nAnswer:" for q in df["question"]
        ]

    else:
        raise ValueError(f"Unsupported task: {task}")


# =========================
# 推理（统一走 generate）
# =========================
# def _ensure_pils(images):
#     pil_list = []
#     for im in images:
#         if isinstance(im, Image.Image):
#             pil_list.append(im)
#         elif isinstance(im, (bytes, bytearray)):
#             from io import BytesIO
#             pil_list.append(Image.open(BytesIO(im)).convert("RGB"))
#         elif isinstance(im, str):
#             if os.path.exists(im):
#                 pil_list.append(Image.open(im).convert("RGB"))
#             else:
#                 raise ValueError(f"Invalid image path: {im}")
#         else:
#             raise TypeError(f"Unsupported image type: {type(im)}")
#     return pil_list
from copy import deepcopy


def _is_minicpm(model_name: str, tokenizer=None, processor=None) -> bool:
    """
    粗判是否 MiniCPM 家族（名称或 tokenizer / processor 线索）。
    """
    name_l = (model_name or "").lower()
    if "minicpm" in name_l:
        return True
    try:
        tok_name = getattr(tokenizer, "name_or_path", "") or getattr(tokenizer, "_name_or_path", "")
        if "minicpm" in str(tok_name).lower():
            return True
    except Exception:
        pass
    try:
        proc_name = getattr(processor, "name_or_path", "") or getattr(processor, "_name_or_path", "")
        if "minicpm" in str(proc_name).lower():
            return True
    except Exception:
        pass
    return False


from copy import deepcopy
from typing import List, Optional, Union
from PIL import Image

from copy import deepcopy
from typing import List, Optional, Union
from PIL import Image
import torch


def _move_to_device(x, device):
    """递归把 batch 里的张量搬到 device；其余类型保持不变。"""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_move_to_device(v, device) for v in x]
        return type(x)(t)  # 保持原有的 list/tuple 类型
    return x  # int/float/str/None/tuple-of-ints 等直接返回


def _build_minicpm_inputs(
        processor,
        images: Optional[Union[Image.Image, List[Image.Image]]],
        user_text: str,
        device,
        max_inp_length: int = 2048,
        system_prompt: Optional[str] = None,
):
    # 1) 归一化 images -> List[Image.Image]
    if images is None:
        img_list: List[Image.Image] = []
    elif isinstance(images, Image.Image):
        img_list = [images]
    else:
        img_list = list(images)

    # 2) 构造消息：首条 user，内容=[多张图..., 文本]
    msgs = [{"role": "user", "content": img_list + [user_text]}]
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}] + msgs

    # 3) 拷贝并把 Image 替换为占位符，收集真正的 images_out
    copy_msgs = deepcopy(msgs)
    images_out: List[Image.Image] = []
    for i, msg in enumerate(copy_msgs):
        role = msg["role"]
        assert role in ["user", "assistant", "system"]
        content = msg["content"]
        if isinstance(content, str):
            content = [content]
        cur_parts = []
        for c in content:
            if isinstance(c, Image.Image):
                images_out.append(c)
                cur_parts.append("(<image>./</image>)")
            elif isinstance(c, str):
                # 防止用户文本里自行放占位符导致重复
                s = c.replace("(<image>./</image>)", "").replace("<image>", "")
                cur_parts.append(s)
        msg["content"] = "\n".join(cur_parts)

    # 4) 生成文本 prompt
    prompt = processor.tokenizer.apply_chat_template(
        copy_msgs, tokenize=False, add_generation_prompt=True
    )

    # 5) 打包模型输入；注意 images 列表单独传
    inputs = processor(
        prompt,
        images_out,
        return_tensors="pt",
        max_length=max_inp_length,
    ).to(device)

    # 6) 递归搬到 device（替换你原来的 {k: v.to(device) for ...}）
    # inputs = _move_to_device(inputs, device)
    return inputs


def detection_prompting(model, tokenizer, processor, sample,
                        candidate_premature_layers, atten_ratio=0.1, ffn_ratio=0.1, model_name=''):
    """
    Run one detection prompt, supporting both text-only and multimodal inputs.
    Returns: hidden_states, generated answer, activate, o_layers
    """
    cut_off_len = 256
    device = next(model.parameters()).device
    kwargs = {}

    # === 多模态样本 ===
    if isinstance(sample, dict) and "images" in sample and "prompt" in sample:
        # ---- 1) 统一为 PIL ----
        def _ensure_pils(images):
            import base64, os
            from io import BytesIO
            from PIL import Image as PILImage
            try:
                import numpy as np
            except Exception:
                np = None
            try:
                import torch
            except Exception:
                torch = None

            def _b64_to_bytes(s: str) -> bytes:
                s = "".join(s.strip().split())
                s += "=" * (-len(s) % 4)
                return base64.b64decode(s)

            def _open_bytes(b: bytes) -> PILImage:
                return PILImage.open(BytesIO(b)).convert("RGB")

            def _str_to_pils(s: str):
                s = s.strip()
                if s.lower().startswith("data:image/"):
                    b64 = s.split(",", 1)[1]
                    return [_open_bytes(_b64_to_bytes(b64))]
                if os.path.exists(s):
                    return [PILImage.open(s).convert("RGB")]
                if s.startswith(("http://", "https://")):
                    import requests
                    r = requests.get(s, timeout=15)
                    r.raise_for_status()
                    return [_open_bytes(r.content)]
                try:
                    return [_open_bytes(_b64_to_bytes(s))]
                except Exception:
                    raise ValueError(f"Invalid image string (not path/url/base64): {s[:80]}...")

            def _np_to_pils(a):
                if np is None:
                    raise TypeError("Got numpy array but numpy is unavailable.")
                import numpy as _np
                if a.ndim == 4:
                    out = []
                    for i in range(a.shape[0]):
                        out += _np_to_pils(a[i])
                    return out
                if a.ndim == 1:
                    if a.dtype == _np.uint8:
                        try:
                            return [_open_bytes(a.tobytes())]
                        except Exception:
                            pass
                    return _list_to_pils(a.tolist())
                if a.ndim == 2:
                    if a.dtype != _np.uint8:
                        if a.dtype.kind in "fc" and a.max() <= 1.0:
                            a = (a * 255.0)
                        a = _np.clip(a, 0, 255).astype(_np.uint8)
                    a = _np.stack([a, a, a], axis=-1)
                if a.ndim == 3:
                    if a.shape[0] in (1, 3, 4) and a.shape[0] != a.shape[-1]:
                        a = a.transpose(1, 2, 0)
                    if a.dtype != _np.uint8:
                        if a.dtype.kind in "fc" and a.max() <= 1.0:
                            a = (a * 255.0)
                        a = _np.clip(a, 0, 255).astype(_np.uint8)
                    return [PILImage.fromarray(a).convert("RGB")]
                raise TypeError(f"Unsupported numpy ndim: {a.ndim}")

            def _torch_to_pils(t):
                if torch is None:
                    raise TypeError("Got torch tensor but torch is unavailable.")
                return _np_to_pils(t.detach().cpu().numpy())

            def _one(x):
                if isinstance(x, PILImage.Image):
                    return [x]
                if isinstance(x, (bytes, bytearray)):
                    return [_open_bytes(bytes(x))]
                if isinstance(x, str):
                    return _str_to_pils(x)
                if np is not None and isinstance(x, np.ndarray):
                    return _np_to_pils(x)
                if torch is not None and isinstance(x, torch.Tensor):
                    return _torch_to_pils(x)
                raise TypeError(f"Unsupported image type: {type(x)}")

            def _list_to_pils(xs):
                out = []
                for e in xs:
                    if isinstance(e, list):
                        out.extend(_list_to_pils(e))
                    else:
                        out.extend(_one(e))
                return out

            if isinstance(images, list):
                return _list_to_pils(images)
            return _one(images)

        images = _ensure_pils(sample["images"])
        text = (sample.get("prompt") or "").strip()

        from PIL import Image

        def preprocess_images(images, max_size=512):
            """Resize images to avoid OOM, keeping aspect ratio."""
            processed = []
            for img in images:
                if isinstance(img, Image.Image):
                    w, h = img.size
                    scale = min(max_size / w, max_size / h, 1.0)  # 只缩小不放大
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                processed.append(img)
            return processed

        if "llava" in str(model_name).lower():
            images = _downscale_images_preserving_aspect(
                images,
                max_side=64,
                max_megapixels_per_image=1.0,
                max_total_megapixels=2.0,
            )
        else:
            # 其它 VLM（MiniCPM、Idefics3 等）也建议做一次温和下采样
            images = _downscale_images_preserving_aspect(
                images,
                max_side=64,
                max_megapixels_per_image=1.5,
                max_total_megapixels=3.0,
            )

        # ... 你已有的 images 下采样之后

        model_dtype = getattr(getattr(model, "dtype", None), "type", None)
        model_dtype = getattr(model, "dtype", torch.float16)  # 兜底

        if _is_minicpm(model_name, tokenizer, processor):
            inputs = _build_minicpm_inputs(
                processor=processor,
                images=images,
                user_text=text,
                device=device,
                max_inp_length=max(768, cut_off_len)  # 稍小点儿，进一步省显存
            )
            # 注意：MiniCPM 这条路直接把 inputs 传给 generate（见 C 段）
        else:
            try:
                proc_inputs = processor(
                    text=[text],  # ← 统一用列表
                    images=[images],  # ← 外层再包一层，保持 batch=1
                    return_tensors="pt",
                    padding=False,
                )
            except TypeError:
                proc_inputs = processor(
                    text=[text],
                    images=[images],  # ← 这里同样要加方括号
                    return_tensors="pt",
                    padding=False,
                )

            # ✅ 关键：半精度+无阻塞搬 GPU
            def _cast_and_move_to_device(batch, device, model_dtype=torch.float16):
                out = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        if v.is_floating_point():
                            out[k] = v.to(device=device, dtype=model_dtype, non_blocking=True)
                        else:
                            out[k] = v.to(device=device, non_blocking=True)  # e.g., input_ids (long)
                    else:
                        out[k] = v
                return out

            proc_inputs = _cast_and_move_to_device(proc_inputs, device, model_dtype=model_dtype)
            for k, v in proc_inputs.items():
                kwargs[k] = v


    # === 文本样本 ===（保留原逻辑）
    elif isinstance(sample, str):
        inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=cut_off_len)
        kwargs['input_ids'] = inputs.input_ids[:, :cut_off_len].to(device)
        kwargs['attention_mask'] = inputs.attention_mask[:, :cut_off_len].to(device)
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    # 生成控制参数（保持原状）
    kwargs.update({
        'max_new_tokens': 1,
        'candidate_premature_layers': candidate_premature_layers,
        'top_ratio_atten': atten_ratio,
        'top_ratio_ffn': ffn_ratio
    })

    # # 自定义 generate
    # # try:
    # hidden_states, outputs, activate, o_layers = model.generate(**kwargs)
    # # except:
    # # hidden_states, outputs, activate, o_layers = model.generate(model_inputs=inputs,  tokenizer=tokenizer,**kwargs)
    # # 收集各层 hidden embedding（保持原状）
    hidden_embed = {}
    # 自定义 generate（推理模式 + 关 cache）
    gen_kwargs = dict(**kwargs)
    try:
        # 有的自定义 generate 支持 use_cache
        gen_kwargs.update(use_cache=False)
    except Exception:
        pass

    with torch.inference_mode():
        if _is_minicpm(model_name, tokenizer, processor):
            # MiniCPM 需要把 build 出来的 inputs 直接传进去，避免内部再构一遍
            try:
                hidden_states, outputs, activate, o_layers = model.generate(model_inputs=inputs, tokenizer=tokenizer,**gen_kwargs)
            except TypeError:
                # 兼容没有 model_inputs 的实现
                gen_kwargs.update(inputs)
                hidden_states, outputs, activate, o_layers = model.generate(**gen_kwargs)
        else:
            hidden_states, outputs, activate, o_layers = model.generate(**gen_kwargs)

    for layer_idx in candidate_premature_layers:
        try:
            hidden_embed[layer_idx] = tokenizer.decode(hidden_states[layer_idx][0])
        except Exception:
            hidden_embed[layer_idx] = str(hidden_states[layer_idx][0].tolist())

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return hidden_embed, answer, activate, o_layers


def save_neuron(activate_neurons, path):
    for group in activate_neurons:
        entry = activate_neurons[group]
        activate_neurons[group] = {key: list(value) if isinstance(value, set) else value
                                   for key, value in entry.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(activate_neurons, f, ensure_ascii=False, indent=2)


def detect_key_neurons(model, tokenizer, processor, lang,
                       atten_ratio=0.1, ffn_ratio=0.1, test_size=-1, candidate_layers=[],
                       detection_path="./test_data/oscar", output_path="./output",
                       suffix="", model_name="", sample_size=10000, task="detect", args=None) -> dict:
    """
    Detects neurons key to the language *lang* and writes results.
    """
    try:
        # 一般 LLM（顶层就有 num_hidden_layers）
        candidate_layers = model.config.num_hidden_layers
    except AttributeError:
        try:
            # 某些模型封装在 llm_config 里
            candidate_layers = model.config.llm_config.num_hidden_layers  # type: ignore
        except AttributeError:
            try:
                # Idefics3 / LLaVA 这类多模态 → 在 text_config 里
                candidate_layers = model.config.text_config.num_hidden_layers  # type: ignore
            except AttributeError:
                print(model.config)
                candidate_layers = None
    candidate_layers = range(int(candidate_layers))
    # 统一从数据加载器拿样本
    lines = load_lines_from_dataset(task, lang, args)
    if sample_size > 0 and sample_size < len(lines):
        lines = random.sample(lines, sample_size)

    activate_key_sets = {
        "fwd_up": [], "fwd_down": [],
        "attn_q": [], "attn_k": [], "attn_v": [], "attn_o": []
    }
    error_count = 0
    intermediate_layers_decode = {}

    print("Detection corpus size: ", len(lines))
    count = 0
    for sample in tqdm(lines):
        # try:
        hidden, answer, activate, o_layers = detection_prompting(
            model, tokenizer, processor, sample, candidate_layers,
            atten_ratio=atten_ratio, ffn_ratio=ffn_ratio, model_name=model_name
        )
        for key in activate.keys():
            activate_key_sets[key].append(activate[key])
        count += 1
        intermediate_layers_decode[count] = hidden
        # except Exception as e:
        #     error_count += 1
        #     count += 1
        #     print(f"[Error #{error_count}] {e}")

    print("Detection query complete; error: ", error_count)

    # 求交集（保持你原逻辑）
    for group in activate_key_sets.keys():
        entries = activate_key_sets[group]
        if not entries:
            activate_key_sets[group] = {}
            continue
        common_layers = {}
        for layer in entries[0].keys():
            if all(layer in d for d in entries):
                arrays = [d[layer] for d in entries]
                common_elements = set.intersection(*map(set, arrays))
                common_layers[layer] = {int(x) for x in common_elements}
        activate_key_sets[group] = common_layers
        print(f"{group} integrated and logged")

    # 文件命名逻辑（保留你的实现）
    if "huggingface" in model_name:
        train_on_lang = "base_model"
        file_name_prefix = model_name.split('/')[-1]
    elif "llama" in model_name.split('/')[-2].lower() or "gemma" in model_name.split('/')[-2].lower():
        train_on_lang = model_name.split('/')[-2][:2]
        file_name_prefix = model_name.split('/')[-2]
    elif "llama" in model_name.split('/')[-1].lower() or "gemma" in model_name.split('/')[-1].lower():
        train_on_lang = model_name.split('/')[-1][:2]
        file_name_prefix = model_name.split('/')[-1]
    else:
        train_on_lang = "base"
        file_name_prefix = model_name.split('/')[-1]

    file_name = f"{file_name_prefix}_{task}_{lang}_atten{atten_ratio}_ffn{ffn_ratio}.json"
    file_path = os.path.join(output_path, file_name)
    save_neuron(activate_key_sets, file_path)

    return activate_key_sets


def detection_all(model_name, lang, atten_ratio=0.1, ffn_ratio=0.1, test_size=-1,
                  detection_path="./corpus_all", output_path="./output",
                  suffix="", sample_size=1000, task="detect", args=None):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    # ============ 模型选择 ============
    name_l = model_name.lower()
    if "gemma-3" in name_l:
        model = Gemma3ForConditionalGeneration.from_pretrained(model_name, config=config, device_map="auto")
        model = model.language_model
    elif "llama" in name_l and "idefics" not in name_l and "minicpm" not in name_l and 'llava' not in name_l:
        import torch
        model = LlamaForCausalLMDetect.from_pretrained(model_name, config=config, device_map="auto")
    elif "gemma-2" in name_l:
        model = Gemma2ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "qwen3" in name_l:
        model = Qwen3ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "qwen" in name_l:
        model = Qwen2ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "gemma" in name_l:
        model = GemmaForCausalLM.from_pretrained(model_name, config=config, device_map="auto")

    # === NEW: MiniCPM-V ===
    elif "minicpm" in name_l or "minicpm-v" in name_l:
        import torch
        model = MiniCPMVForCausalLM.from_pretrained(model_name, config=config, device_map="cuda:0",torch_dtype=torch.float16)  # NEW

    # === NEW: Idefics3 ===
    elif "idefics3" in name_l or "idefics-3" in name_l:
        import torch
        model = Idefics3ForConditionalGeneration.from_pretrained(model_name, config=config, device_map="auto",torch_dtype=torch.float16) # NEW

    elif "llava" in name_l:
        import torch
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, config=config, device_map="auto",
                                                                  torch_dtype=torch.float16)  # NEW , device_map="auto"

    else:
        raise ValueError(f"Model {model_name} not supported")

    for l in lang:
        print("Detecting neurons for", l)
        neurons = detect_key_neurons(
            model, tokenizer, processor, l,
            atten_ratio=atten_ratio, ffn_ratio=ffn_ratio,
            test_size=test_size, detection_path=detection_path, output_path=output_path,
            suffix=suffix, model_name=model_name, sample_size=sample_size, task=task, args=args
        )
        # 这里原来打印 len(neurons['attn_q'][0]) 可能报错，略微调整：
        attn_q_layers = neurons.get("attn_q", {})
        print(l, "complete", len(attn_q_layers.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training / Detection args
    parser.add_argument("--corpus_path", type=str, default='./')
    parser.add_argument("--corpus_size", type=int, default=-1)
    parser.add_argument("--base", type=str,
                        default="openbmb/MiniCPM-Llama3-V-2_5")  # openbmb/MiniCPM-Llama3-V-2_5MergeBench/Llama-3.1-8B_math HuggingFaceM4/Idefics3-8B-Llama3  llava-hf/llama3-llava-next-8b-hf
    parser.add_argument("--output_path", type=str, default="./neuron_train_data_detect_with_prediction/")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--atten_ratio", type=float, default=0.2)
    parser.add_argument("--ffn_ratio", type=float, default=0.2)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--task", type=str, default="multimodal")
    # NEW: multimodal 数据文件（parquet）
    parser.add_argument("--multimodal_file", type=str, default='minicpm_vl_math.parquet')  # NEW

    args = parser.parse_args()

    if ',' in args.lang:
        lang_set = args.lang.split(',')
    else:
        lang_set = [args.lang]

    detection_all(
        args.base, lang_set,
        args.atten_ratio, args.ffn_ratio,
        args.corpus_size, args.corpus_path,
        args.output_path, args.suffix,
        args.sample_size, args.task, args
    )
