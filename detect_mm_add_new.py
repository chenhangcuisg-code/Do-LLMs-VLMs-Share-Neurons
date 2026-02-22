import os, sys, json, random, argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from transformers import AutoTokenizer, AutoConfig, AutoProcessor

# Custom models
from transformers_custom.modeling_llama import LlamaForCausalLMDetect
from transformers_custom.modeling_minicpmv import MiniCPMV as MiniCPMVForCausalLM
from transformers_custom.modeling_idefics import Idefics3ForConditionalGeneration
from transformers_custom.modeling_llava_next import LlavaNextForConditionalGeneration
try:
    from transformers_custom.modeling_qwen2 import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = None

def _downscale_images_preserving_aspect(
    images,
    max_side: int = 1024,
    max_megapixels_per_image: float = 1.0,
    max_total_megapixels: float = 2.0,
    resample = Image.BICUBIC
):
    """
    Downscale PIL images while preserving aspect ratio.
    Limits: max side per image, max megapixels per image, max total megapixels.
    """
    def _shrink_to(image: Image.Image, target_mp: float, max_side_px: int) -> Image.Image:
        w, h = image.size
        scale_side = min(1.0, max_side_px / max(w, h)) if max(w, h) > max_side_px else 1.0
        cur_mp = (w * h) / 1_000_000.0
        scale_mp = (target_mp / cur_mp) ** 0.5 if cur_mp > target_mp else 1.0
        scale = min(scale_side, scale_mp)
        if scale < 1.0:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            image = image.resize((nw, nh), resample=resample)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    imgs = [ _shrink_to(im, max_megapixels_per_image, max_side) for im in images ]

    total_pixels = sum(im.size[0] * im.size[1] for im in imgs)
    total_mp = total_pixels / 1_000_000.0
    if total_mp > max_total_megapixels and total_pixels > 0:
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
# Data loading (multimodal)
# =========================
def load_lines_from_dataset(task, lang, args):
    """
    Load dataset lines or (image(s), text) samples.
    Supports:
      - 'multimodal' -> Parquet with problem, answer (optional), images (paths or base64)
      - 'gsm' -> llama3ds_math.tsv
    Returns unified sample structure.
    """
    if task == "multimodal":
        if not getattr(args, "multimodal_file", None):
            raise ValueError("Provide --multimodal_file pointing to Parquet with problem, answer, images columns.")
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
# Inference (unified via generate)
# =========================
from copy import deepcopy


def _is_minicpm(model_name: str, tokenizer=None, processor=None) -> bool:
    """Check if model is MiniCPM family."""
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
import torch


def _move_to_device(x, device):
    """Recursively move tensors in batch to device; other types unchanged."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_move_to_device(v, device) for v in x]
        return type(x)(t)
    return x


def _build_minicpm_inputs(
        processor,
        images: Optional[Union[Image.Image, List[Image.Image]]],
        user_text: str,
        device,
        max_inp_length: int = 2048,
        system_prompt: Optional[str] = None,
):
    if images is None:
        img_list: List[Image.Image] = []
    elif isinstance(images, Image.Image):
        img_list = [images]
    else:
        img_list = list(images)

    msgs = [{"role": "user", "content": img_list + [user_text]}]
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}] + msgs

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
                s = c.replace("(<image>./</image>)", "").replace("<image>", "")
                cur_parts.append(s)
        msg["content"] = "\n".join(cur_parts)

    prompt = processor.tokenizer.apply_chat_template(
        copy_msgs, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        prompt,
        images_out,
        return_tensors="pt",
        max_length=max_inp_length,
    ).to(device)

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

    if isinstance(sample, dict) and "images" in sample and "prompt" in sample:
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
                    raise ValueError(f"Invalid image string: {s[:80]}...")

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
                    scale = min(max_size / w, max_size / h, 1.0)
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
            images = _downscale_images_preserving_aspect(
                images,
                max_side=64,
                max_megapixels_per_image=1.5,
                max_total_megapixels=3.0,
            )

        model_dtype = getattr(getattr(model, "dtype", None), "type", None)
        model_dtype = getattr(model, "dtype", torch.float16)

        if _is_minicpm(model_name, tokenizer, processor):
            inputs = _build_minicpm_inputs(
                processor=processor,
                images=images,
                user_text=text,
                device=device,
                max_inp_length=max(768, cut_off_len)
            )
        else:
            try:
                proc_inputs = processor(
                    text=[text],
                    images=[images],
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
                            out[k] = v.to(device=device, non_blocking=True)
                    else:
                        out[k] = v
                return out

            proc_inputs = _cast_and_move_to_device(proc_inputs, device, model_dtype=model_dtype)
            for k, v in proc_inputs.items():
                kwargs[k] = v


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
            try:
                hidden_states, outputs, activate, o_layers = model.generate(model_inputs=inputs, tokenizer=tokenizer,**gen_kwargs)
            except TypeError:
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
    """Detect neurons key to the given language and write results."""
    try:
        candidate_layers = model.config.num_hidden_layers
    except AttributeError:
        try:
            candidate_layers = model.config.llm_config.num_hidden_layers
        except AttributeError:
            try:
                candidate_layers = model.config.text_config.num_hidden_layers
            except AttributeError:
                print(model.config)
                candidate_layers = None
    candidate_layers = range(int(candidate_layers))
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
        hidden, answer, activate, o_layers = detection_prompting(
            model, tokenizer, processor, sample, candidate_layers,
            atten_ratio=atten_ratio, ffn_ratio=ffn_ratio, model_name=model_name
        )
        for key in activate.keys():
            activate_key_sets[key].append(activate[key])
        count += 1
        intermediate_layers_decode[count] = hidden

    print("Detection query complete; error: ", error_count)

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
    name_l = model_name.lower()
    if "gemma-3" in name_l:
        from transformers_custom.modeling_gemma3 import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(model_name, config=config, device_map="auto")
        model = model.language_model
    elif "llama" in name_l and "idefics" not in name_l and "minicpm" not in name_l and 'llava' not in name_l:
        import torch
        model = LlamaForCausalLMDetect.from_pretrained(model_name, config=config, device_map="auto")
    elif "gemma-2" in name_l:
        from transformers_custom.modeling_gemma2 import Gemma2ForCausalLM
        model = Gemma2ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "qwen3" in name_l:
        from transformers_custom.modeling_qwen3 import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "qwen" in name_l and "vl" in name_l:
        from transformers_custom.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        import torch
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, config=config, device_map="auto", torch_dtype=torch.float16)
    elif "qwen" in name_l and Qwen2ForCausalLM is not None:
        model = Qwen2ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "gemma" in name_l:
        from transformers_custom.modeling_gemma import GemmaForCausalLM
        model = GemmaForCausalLM.from_pretrained(model_name, config=config, device_map="auto")

    elif "minicpm" in name_l or "minicpm-v" in name_l:
        import torch
        model = MiniCPMVForCausalLM.from_pretrained(model_name, config=config, device_map="cuda:0",torch_dtype=torch.float16)

    elif "idefics3" in name_l or "idefics-3" in name_l:
        import torch
        model = Idefics3ForConditionalGeneration.from_pretrained(model_name, config=config, device_map="auto",torch_dtype=torch.float16)

    elif "llava" in name_l:
        import torch
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, config=config, device_map="auto",
                                                                  torch_dtype=torch.float16)

    elif "internvl" in name_l:
        from transformers_custom.modeling_internvl_chat import InternVLChatModel
        import torch
        model = InternVLChatModel.from_pretrained(model_name, config=config, device_map="auto", torch_dtype=torch.float16)

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
        attn_q_layers = neurons.get("attn_q", {})
        print(l, "complete", len(attn_q_layers.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default='./')
    parser.add_argument("--corpus_size", type=int, default=-1)
    parser.add_argument("--base", type=str,
                        default="openbmb/MiniCPM-Llama3-V-2_5")
    parser.add_argument("--output_path", type=str, default="./neuron_train_data_detect_with_prediction/")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--atten_ratio", type=float, default=0.2)
    parser.add_argument("--ffn_ratio", type=float, default=0.2)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--task", type=str, default="multimodal")
    parser.add_argument("--multimodal_file", type=str, default='minicpm_vl_math.parquet')

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
