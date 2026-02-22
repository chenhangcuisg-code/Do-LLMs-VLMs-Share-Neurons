#!/usr/bin/env python
# coding: utf-8
"""
Batch inference entry point for text-only and multimodal math tasks.

Supported models/backends
-------------------------
- Multimodal:  MiniCPM-Llama3-V-2.5   (backend: ``minicpm``)
- Multimodal:  Idefics3               (backend: ``idefics3``)
- Multimodal:  LLaVA-Next             (backend: ``llava_next``)
- Text-only:   Llama 3 Instruct       (backend: ``llama3``)

Tasks
-----
- ``multimodal``: use Geometry3K (with images) and call a multimodal backend.
- ``gsm`` / ``gsm_mm``: read ``mgsm_{lang}.tsv`` and call a text-only backend.

Outputs
-------
- ``task`` in {``gsm``, ``gsm_mm``}: TSV (``question \t prediction``)
- ``multimodal``: Parquet with prediction, problem, answer, and base64-encoded images.
"""

import argparse
import base64
import io
import os
from typing import Any, Dict, List

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)

# =========================
# Data loading
# =========================

def load_lines_from_dataset(task: str, lang: str) -> List[Dict[str, Any]]:
    if task == 'multimodal':
        ds = load_dataset('hiyouga/geometry3k')
        full = concatenate_datasets([ds['train'], ds['validation'], ds['test']])
        samples = []
        for ex in full:
            samples.append({
                'problem': ex['problem'],
                'answer': ex['answer'],
                'images': ex['images'],
                'prompt': ex['problem'],
            })
        return samples

    elif task in ('gsm', 'gsm_mm'):
        file_path = f'mgsm_{lang}.tsv'
        df = pd.read_csv(file_path, sep='\t', names=['question', 'answer'])
        return [{'question': q, 'answer': a, 'prompt': q} for q, a in zip(df['question'], df['answer'])]

    else:
        raise ValueError(f'Unsupported task: {task}')


# =========================
# Image processing utilities
# =========================

def to_pil(img_like: Any) -> Image.Image:
    if isinstance(img_like, Image.Image):
        return img_like.convert('RGB')
    if isinstance(img_like, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_like)).convert('RGB')
    if isinstance(img_like, str):
        s = img_like.strip()
        if s.startswith('data:image'):
            b64 = s.split(',', 1)[-1]
            data = base64.b64decode(b64)
            return Image.open(io.BytesIO(data)).convert('RGB')
        try:
            data = base64.b64decode(s, validate=True)
            return Image.open(io.BytesIO(data)).convert('RGB')
        except Exception:
            pass
        if os.path.isfile(s):
            return Image.open(s).convert('RGB')
    raise ValueError("Unsupported image type for to_pil().")


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =========================
# Llama3 template fallback
# =========================

def has_chat_template(obj: Any) -> bool:
    try:
        tmpl = getattr(obj, "chat_template", None)
        return isinstance(tmpl, str) and len(tmpl.strip()) > 0
    except Exception:
        return False


def ensure_llama3_fallback_chat_template(tokenizer) -> None:
    if has_chat_template(tokenizer):
        return
    tokenizer.chat_template = (
        "{{ bos_token if bos_token is defined else '' }}"
        "{% for message in messages %}"
        "{% set role = message['role'] %}"
        "{% set content = message['content'] %}"
        "{% if role == 'system' %}System: {{ content }}\n"
        "{% elif role == 'user' %}User: {{ content }}\n"
        "{% elif role == 'assistant' %}Assistant: {{ content }}\n"
        "{% else %}{{ role|capitalize }}: {{ content }}\n{% endif %}"
        "{% endfor %}"
        "Assistant:"
    )


# =========================
# Inferencer
# =========================

DEFAULT_MODELS = {
    "minicpm": "openbmb/MiniCPM-Llama3-V-2_5",
    "idefics3": "HuggingFaceM4/Idefics3-8B-Llama3",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llava_next": "llava-hf/llama3-llava-next-8b-hf",
}



class Inferencer:
    def __init__(
        self,
        task: str,
        backend: str,
        model_path: str = None,
        device: str = None,
        bs: int = 4,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.2,
    ):
        self.task = task
        self.backend = backend
        self.bs = bs
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model_name = model_path or DEFAULT_MODELS[backend]

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None

        if backend == "minicpm":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            )
            self.model = self.model.to(device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.processor = None
            self.model.eval()
        elif backend == "idefics3":
            # Use Vision2Seq version
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16
            ).to(self.device)

        elif backend == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
            ensure_llama3_fallback_chat_template(self.tokenizer)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            self.processor = None
        elif backend == "llava_next":
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            # FP16 + auto device placement
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                device_map="cuda:1" if torch.cuda.is_available() else None,
            )
            self.tokenizer = None

        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.model.eval()

    @torch.no_grad()
    def infer_multimodal_batch(self, batch):
        if self.backend == "minicpm":
            preds = []
            for s in batch:
                # 1) Load image(s)
                imgs = [to_pil(x) for x in s['images']]
                image_arg = imgs[0] if len(imgs) == 1 else imgs

                # 2) Clean any image placeholders in prompt
                import re
                prompt_text = s['prompt']
                prompt_text = re.sub(r'</?\s*image\s*>', ' ', prompt_text, flags=re.I)
                prompt_text = re.sub(r'\[/?\s*image\s*\]', ' ', prompt_text, flags=re.I)
                prompt_text = re.sub(r'</?\s*img\s*>', ' ', prompt_text, flags=re.I)
                prompt_text = re.sub(r'\s+', ' ', prompt_text).strip()

                msgs = [{'role': 'user', 'content': prompt_text}]
                print(msgs)
                # 3) Call MiniCPM official interface
                res = self.model.chat(
                    image=image_arg,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    sampling=bool(self.do_sample or True),
                    temperature=float(self.temperature),
                )
                preds.append(res.strip() if isinstance(res, str) else str(res).strip())
            return preds
        if self.backend == "llava_next":
            preds = []
            convs = []
            images_batch = []
            for s in batch:
                # Multi-image: generate {"type":"image"} per image
                imgs = [to_pil(x) for x in s['images']]
                content = [{"type": "text", "text": s['prompt'].replace("<image>", "").strip()}]
                for _ in imgs:
                    content.append({"type": "image"})
                convs.append([{"role": "user", "content": content}])

                # images param: single PIL or list for multi-image
                images_batch.append(imgs[0] if len(imgs) == 1 else imgs)

            prompts = [
                self.processor.apply_chat_template(c, add_generation_prompt=True)
                for c in convs
            ]

            inputs = self.processor(
                images=images_batch,
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
            }
            # Sampling settings (aligned with main class)
            if self.do_sample:
                gen_kwargs.update(dict(do_sample=True, temperature=float(self.temperature)))

            outputs = self.model.generate(**inputs, **gen_kwargs)
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
            return [d.strip() for d in decoded]


        # Other backends (e.g. idefics3) use self.processor.apply_chat_template
        convs = []
        image_lists = []
        for s in batch:
            imgs = [to_pil(x) for x in s['images']]
            content = [{"type": "image"} for _ in imgs]
            content.append({"type": "text", "text": f"{s['prompt'].replace('<image>', '')}"})
            convs.append([{"role": "user", "content": content}])
            image_lists.append(imgs)

        prompt = [self.processor.apply_chat_template(c, add_generation_prompt=True) for c in convs]
        inputs = self.processor(text=prompt, images=image_lists, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        preds = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [p.strip() for p in preds]

    # ---------- Text inference ----------
    @torch.no_grad()
    def infer_text_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        messages_list = []
        for s in batch:
            messages = [
                # {"role": "system", "content": "You are a reliable math assistant. Give the final answer in <answer>...</answer>."},
                {"role": "user", "content": s['prompt']},
            ]
            messages_list.append(messages)

        texts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], outputs)]
        preds = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)
        return [p.strip() for p in preds]

    # ---------- Unified entry ----------
    @torch.no_grad()
    def batch_infer(self, batch: List[Dict[str, Any]]) -> List[str]:
        if self.task == 'multimodal':
            return self.infer_multimodal_batch(batch)
        elif self.task in ('gsm', 'gsm_mm'):
            return self.infer_text_batch(batch)
        else:
            raise ValueError(f"Unknown task: {self.task}")


# =========================
# Main entry
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['multimodal', 'gsm', 'gsm_mm'], default='gsm')
    parser.add_argument('--backend', choices=['minicpm', 'idefics3', 'llama3', 'llava_next'], default='llama3')
    parser.add_argument('--model_path', default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B')#MathGenie/MathCoder2-Llama-3-8B 'llava-hf/llama3-llava-next-8b-hf
    parser.add_argument('--lang', default='en')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out_file', default='llama3ds_math.tsv')
    args = parser.parse_args()

    samples = load_lines_from_dataset(args.task, args.lang)

    infer = Inferencer(
        task=args.task,
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        bs=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )

    all_results = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc='Inferencing'):
        batch = samples[i: i + args.batch_size]
        preds = infer.batch_infer(batch)

        for s, p in zip(batch, preds):
            rec = {'prediction': p.strip()}
            if args.task == 'multimodal':
                img_b64 = [image_to_base64(to_pil(it)) for it in s['images']]
                rec.update({
                    'problem': s['problem'],
                    'answer': s['answer'],
                    'images': img_b64,
                    'prompt': s['prompt']
                })
            else:
                rec.update({'question': s['prompt'], 'answer': s['answer']})
            all_results.append(rec)

    df = pd.DataFrame(all_results)
    if args.task in ('gsm', 'gsm_mm'):
        out_tsv = args.out_file if args.out_file.endswith('.tsv') else os.path.splitext(args.out_file)[0] + '.tsv'
        df[['question', 'answer']].to_csv(out_tsv, sep='\t', index=False, header=False)
        print(f"[Saved] {out_tsv}")
    else:
        out_parquet = args.out_file if args.out_file.endswith('.parquet') else os.path.splitext(args.out_file)[0] + '.parquet'
        df.to_parquet(out_parquet, index=False)
        print(f"[Saved] {out_parquet}")


if __name__ == '__main__':
    main()
