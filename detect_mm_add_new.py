import argparse
import json
import os
import random
from copy import deepcopy
from typing import List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from transformers_custom.modeling_llama import LlamaForCausalLMDetect
from transformers_custom.modeling_minicpmv import MiniCPMV as MiniCPMVForCausalLM
from transformers_custom.modeling_idefics import Idefics3ForConditionalGeneration
from transformers_custom.modeling_llava_next import LlavaNextForConditionalGeneration

def _downscale_images_preserving_aspect(
    images,
    max_side: int = 1024,
    max_megapixels_per_image: float = 1.0,
    max_total_megapixels: float = 2.0,
    resample=Image.BICUBIC,
):
    """
    Downscale a list of PIL images while preserving aspect ratio.

    This constrains:
      - maximum side length per image,
      - maximum megapixels per image,
      - total megapixels across all images in one sample.
    """
    def _shrink_to(image: Image.Image, target_mp: float, max_side_px: int) -> Image.Image:
        w, h = image.size
        # First by max side
        scale_side = min(1.0, max_side_px / max(w, h)) if max(w, h) > max_side_px else 1.0
        # Then by MP limit
        cur_mp = (w * h) / 1_000_000.0
        scale_mp = (target_mp / cur_mp) ** 0.5 if cur_mp > target_mp else 1.0
        scale = min(scale_side, scale_mp)
        if scale < 1.0:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            image = image.resize((nw, nh), resample=resample)
        # Convert to RGB, drop alpha/EXIF to save memory
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    # First apply per-image constraints.
    imgs = [_shrink_to(im, max_megapixels_per_image, max_side) for im in images]

    # Then enforce a global constraint across all images if needed.
    total_pixels = sum(im.size[0] * im.size[1] for im in imgs)
    total_mp = total_pixels / 1_000_000.0
    if total_mp > max_total_megapixels and total_pixels > 0:
        # Uniformly downscale again with a sqrt-based ratio.
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

def load_lines_from_dataset(task, lang, args):
    """
    Load dataset lines or (image(s), text) samples.
    Supports:
      - 'multimodal'   -> Parquet with columns: problem, answer (optional), images (list of paths or base64)
      - 'gsm'          -> mgsm_en.tsv

    Returns a unified sample structure:
      - Text task: str (prompt text)
      - Multimodal task: dict { "images": List[str|PIL|bytes], "prompt": str, "answer": Optional[str] }
    """
    if task == "multimodal":
        if not getattr(args, "multimodal_file", None):
            raise ValueError(
                "Please provide --multimodal_file pointing to a Parquet file "
                "containing 'problem', 'answer', and 'images' columns."
            )
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
        df = pd.read_csv(file_path, sep="\t", names=["question", "answer"])
        return [f"Question: {q}\nAnswer:" for q in df["question"]]

    else:
        raise ValueError(f"Unsupported task: {task}")


# =========================
# Inference (unified via generate)
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
def _is_minicpm(model_name: str, tokenizer=None, processor=None) -> bool:
    """
    Heuristically check whether the model belongs to the MiniCPM family
    based on the model name, tokenizer, or processor hints.
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


def _move_to_device(x, device):
    """Recursively move tensors in a batch to the given device."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_move_to_device(v, device) for v in x]
        return type(x)(t)  # Preserve list/tuple type
    return x  # int/float/str/None/tuple-of-ints etc. return as-is


def _build_minicpm_inputs(
        processor,
        images: Optional[Union[Image.Image, List[Image.Image]]],
        user_text: str,
        device,
        max_inp_length: int = 2048,
        system_prompt: Optional[str] = None,
):
    # 1) Normalize images -> List[Image.Image]
    if images is None:
        img_list: List[Image.Image] = []
    elif isinstance(images, Image.Image):
        img_list = [images]
    else:
        img_list = list(images)

    # 2) Build the message list: the first user message contains [images..., text].
    msgs = [{"role": "user", "content": img_list + [user_text]}]
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}] + msgs

    # 3) Copy and replace Image objects with placeholders, collecting real images_out.
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
                # Avoid duplicate placeholders inside user text.
                s = c.replace("(<image>./</image>)", "").replace("<image>", "")
                cur_parts.append(s)
        msg["content"] = "\n".join(cur_parts)

    # 4) Build the textual prompt.
    prompt = processor.tokenizer.apply_chat_template(
        copy_msgs, tokenize=False, add_generation_prompt=True
    )

    # 5) Pack model inputs; note that images are passed separately.
    inputs = processor(
        prompt,
        images_out,
        return_tensors="pt",
        max_length=max_inp_length,
    ).to(device)

    # 6) Recursively move to device
    # inputs = _move_to_device(inputs, device)
    return inputs


def detection_prompting(model, tokenizer, processor, sample,
                        candidate_premature_layers, atten_ratio=0.1, ffn_ratio=0.1, model_name=''):
    """
    Run one detection prompt, supporting both text-only and multimodal inputs.

    Returns
    -------
    hidden_states : dict
        Per-layer hidden states as returned by the model.
    answer : str
        Decoded model answer.
    activate : dict
        Activation statistics for selected neuron groups.
    o_layers : Any
        Additional per-layer outputs from the model.
    """
    cut_off_len = 256
    device = next(model.parameters()).device
    kwargs = {}

    # === Multimodal samples ===
    if isinstance(sample, dict) and "images" in sample and "prompt" in sample:
        # ---- 1) Normalize to PIL images ----
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
                    scale = min(max_size / w, max_size / h, 1.0)  # Downscale only
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
            # Other VLMs (MiniCPM, Idefics3 etc.) also benefit from a light downsampling.
            images = _downscale_images_preserving_aspect(
                images,
                max_side=64,
                max_megapixels_per_image=1.5,
                max_total_megapixels=3.0,
            )

        # After downsampling images

        model_dtype = getattr(getattr(model, "dtype", None), "type", None)
        model_dtype = getattr(model, "dtype", torch.float16)

        if _is_minicpm(model_name, tokenizer, processor):
            inputs = _build_minicpm_inputs(
                processor=processor,
                images=images,
                user_text=text,
                device=device,
                max_inp_length=max(768, cut_off_len),
            )
        else:
            try:
                proc_inputs = processor(
                    text=[text],
                    images=[images],  # Outer wrap for batch=1
                    return_tensors="pt",
                    padding=False,
                )
            except TypeError:
                proc_inputs = processor(
                    text=[text],
                    images=[images],
                    return_tensors="pt",
                    padding=False,
                )

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
            kwargs.update(proc_inputs)

    # === Text-only samples ===
    elif isinstance(sample, str):
        inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=cut_off_len)
        kwargs['input_ids'] = inputs.input_ids[:, :cut_off_len].to(device)
        kwargs['attention_mask'] = inputs.attention_mask[:, :cut_off_len].to(device)
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    kwargs.update({
        'max_new_tokens': 1,
        'candidate_premature_layers': candidate_premature_layers,
        'top_ratio_atten': atten_ratio,
        'top_ratio_ffn': ffn_ratio
    })

    hidden_embed = {}
    gen_kwargs = dict(**kwargs)
    try:
        gen_kwargs.update(use_cache=False)
    except Exception:
        pass

    with torch.inference_mode():
        if _is_minicpm(model_name, tokenizer, processor):
            # MiniCPM needs pre-built inputs to avoid internal reconstruction
            try:
                hidden_states, outputs, activate, o_layers = model.generate(model_inputs=inputs, tokenizer=tokenizer,**gen_kwargs)
            except TypeError:
                # Fallback when model_inputs not available
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
        # Generic LLM (num_hidden_layers at top level)
        candidate_layers = model.config.num_hidden_layers
    except AttributeError:
        try:
            # Some models wrap in llm_config
            candidate_layers = model.config.llm_config.num_hidden_layers  # type: ignore
        except AttributeError:
            try:
                # Idefics3 / LLaVA etc. multimodal -> in text_config
                candidate_layers = model.config.text_config.num_hidden_layers  # type: ignore
            except AttributeError:
                print(model.config)
                candidate_layers = None
    candidate_layers = range(int(candidate_layers))
    # Load samples from data loader
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

    # Compute intersection
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

    # File naming logic
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
    # ============ Model selection ============
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
        # Adjust for safe access to neurons['attn_q'][0]:
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
    # Multimodal data file (parquet)
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
