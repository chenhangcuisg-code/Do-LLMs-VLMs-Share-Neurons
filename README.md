## Overview

This directory contains the code used for multimodal/text-only math reasoning experiments and neuron-level model merging for ICLR-style submissions.
It is organized around three main entry points:

- `detect_mm_add_new.py`: detect important neurons for a given base model on text or multimodal data.
- `merge_smart.py`: merge shared neurons from two source models into a base model with several fusion strategies.
- `inference_mm_new_model.py`: run batched inference for GSM-style text tasks and Geometry3K-style multimodal tasks.

Custom model components are implemented in the `transformers_custom` package, which extends Hugging Face Transformers for specific model families (MiniCPM, Idefics3, LLaVA-Next, Qwen, Gemma, InternVL, etc.).

## Environment and Dependencies

The code assumes:

- Python 3.9+ (3.10 recommended)
- PyTorch with CUDA support (for GPU inference/merging)
- Hugging Face `transformers`, `datasets`
- `pandas`, `tqdm`, `Pillow`

Example installation:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or your CUDA version
pip install transformers datasets pandas tqdm pillow
```

You should also install any additional model-specific dependencies required by the model repositories you use on Hugging Face Hub (see their READMEs).

## 1. Neuron Detection (`detect_mm_add_new.py`)

This script detects language- or task-specific neurons for a given model using either text-only (GSM-style) or multimodal data.
It supports several vision-language models through the `transformers_custom` implementations (MiniCPM-V, Idefics3, LLaVA-Next, etc.).

### Input data

- **Multimodal** (`--task multimodal`):
  - A Parquet file specified by `--multimodal_file`.
  - Required columns:
    - `problem`: text of the problem
    - `answer` (optional): reference answer
    - `images`: list of image paths or base64-encoded images
- **GSM-style text** (`--task gsm`):
  - A TSV file `llama3ds_math.tsv` in `--corpus_path`, with columns:
    - `question`, `answer`

### Key arguments

- `--base`: Hugging Face model id or local path of the base model.
- `--lang`: language code (currently only `"en"` for GSM).
- `--task`: `"multimodal"` or `"gsm"`.
- `--multimodal_file`: Parquet file path for multimodal detection.
- `--output_path`: directory to store JSON files with detected neurons.
- `--sample_size`: number of samples to use for detection (subsampled if larger than dataset size).
- `--atten_ratio`, `--ffn_ratio`: ratios controlling which neurons are treated as "key".

### Example (multimodal detection)

```bash
python detect_mm_add_new.py \
  --base openbmb/MiniCPM-Llama3-V-2_5 \
  --task multimodal \
  --multimodal_file minicpm_vl_math.parquet \
  --lang en \
  --sample_size 100 \
  --atten_ratio 0.2 \
  --ffn_ratio 0.2 \
  --output_path ./neuron_train_data_detect_with_prediction
```

The script writes JSON files containing sets of "key" neurons for each group (e.g. attention Q/K/V/O, FFN up/down).

## 2. Neuron Merging (`merge_smart.py`)

`merge_smart.py` loads three models:

- a **base model** which will be updated in-place,
- **model A**,
- **model B**,

and a JSON file describing "shared" neurons between models, then merges only those neurons into the base model using one of several fusion strategies:

- `mean`: element-wise mean of the corresponding weights.
- `lora`: LoRA-style low-rank deltas on the full projection matrices.

The script includes a `smart_load_model` helper that uses configuration heuristics to select the correct class for different families (Qwen2.5-VL, InternVL2/2.5, LLaVA-Next, other VL/LM models).

### Required JSON format

The neuron JSON must follow the structure produced by `detect_mm_add_new.py`, for example:

```json
{
  "attn_q": {
    "0": [0, 1, 2],
    "1": [5, 6]
  },
  "fwd_up": {
    "0": [10, 11]
  }
}
```

Each top-level key corresponds to a projection group (e.g. `attn_q`, `attn_k`, `attn_v`, `attn_o`, `fwd_up`, `fwd_down`), and each nested dict maps a layer index (as string) to a list of neuron indices.

### Key arguments

- `--base`: base model path/id (weights will be updated and saved).
- `--model_a`: path/id for model A.
- `--model_b`: path/id for model B.
- `--json`: JSON file with shared neuron indices.
- `--output_dir`: output directory for the merged model.
- `--merge_method`: one of `mean`, `lora`.
- `--pca_k`, `--pca_alpha`: PCA mode hyperparameters.
- `--lora_rank`, `--lora_beta`: LoRA mode hyperparameters.
- `--dtype`: numeric precision (`float16`, `bfloat16`, `float32`).
- `--device`: device spec or `"auto"` for automatic device mapping.

### Example (LoRA-based merge)

```bash
python merge_smart.py \
  --base llava-hf/llama3-llava-next-8b-hf \
  --model_a llava-hf/llama3-llava-next-8b-hf \
  --model_b deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --json ./shared_VL_reasoning.json \
  --output_dir ./shared_merged_model_llavanext_lora_merge \
  --merge_method lora \
  --lora_rank 8 \
  --lora_beta 1.0 \
  --dtype float16 \
  --device auto
```

The merged model is saved with `save_pretrained` under `--output_dir`.

### Run All Experiments

A script `run_all_merge_experiments.sh` (or `run_all_merge_experiments.ps1` on Windows) runs the **full pipeline**: neuron detection → compute shared neurons → merge.

| Model | lora_rank | lora_beta | atten_ratio | ffn_ratio |
|-------|-----------|-----------|-------------|-----------|
| Idefics3 | 4 | 0.5 | 0.3 | 0.3 |
| InternVL 3B | 32 | 0.5 | 0.3 | 0.3 |
| LLaVA Next | 4 | 0.5 | 0.3 | 0.3 |
| QwenVL 7B | 4 | 0.5 | 0.1 | 0.1 |
| QwenVL 3B | 4 | 0.5 | 0.1 | 0.1 |
| MiniCPM-V | 4 | 0.5 | 0.3 | 0.3 |

**Environment variables:**
- `DETECT_SCRIPT`: path to `detect_mm_add_new.py` (default: same directory as script)
- `NEURON_BASE`: output dir for detection JSONs and `compare_*/shared_VL_reasoning.json`
- `OUTPUT_BASE`: output dir for merged models
- `CORPUS_PATH`: directory containing `llama3ds_math.tsv` (for GSM detection)
- `MULTIMODAL_FILE`: parquet path for multimodal detection (default: `minicpm_vl_math.parquet`)
- `SAMPLE_SIZE`: samples per detection (default: 100)
- `SKIP_DETECT=1`: skip detection, run only merge (requires existing JSONs)

```bash
CORPUS_PATH=/path/to/corpus MULTIMODAL_FILE=geometry3k.parquet bash run_all_merge_experiments.sh
```

To skip detection and run only merge (when JSONs already exist):

```bash
SKIP_DETECT=1 NEURON_BASE=/path/to/neuron_outputs bash run_all_merge_experiments.sh
```

## 3. Batched Inference (`inference_mm_new_model.py`)

This script provides a unified interface for batched inference on:

- GSM-style text tasks (`task` in `{gsm, gsm_mm}`) using a Llama-3-style causal LM backend.
- Geometry3K-style multimodal tasks using MiniCPM, Idefics3, or LLaVA-Next backends.

The `Inferencer` class handles backend-specific preprocessing:

- MiniCPM: uses its official `.chat` interface with image + text messages.
- Idefics3: uses `AutoModelForVision2Seq` and `AutoProcessor` with `apply_chat_template`.
- LLaVA-Next: uses `LlavaNextForConditionalGeneration` and `LlavaNextProcessor`.
- Llama3: uses `AutoModelForCausalLM` with a fallback chat template when none is provided.

### Key CLI arguments

- `--task`: one of `multimodal`, `gsm`, `gsm_mm`.
- `--backend`: one of `minicpm`, `idefics3`, `llama3`, `llava_next`.
- `--model_path`: Hugging Face model id or local path (overrides backend defaults).
- `--lang`: language code (for GSM tasks, e.g. `en`).
- `--batch_size`: batch size for inference.
- `--max_new_tokens`: maximum new tokens per sample.
- `--do_sample`, `--temperature`: sampling configuration.
- `--device`: device string, e.g. `cuda:0` (or CPU).
- `--out_file`: output TSV or Parquet file.

### Input data conventions

- For `task in {'gsm', 'gsm_mm'}`:
  - Expect a TSV file `mgsm_{lang}.tsv` in the working directory, with `question` and `answer` columns.
- For `task == 'multimodal'`:
  - Geometry3K is loaded from Hugging Face via `datasets`, then concatenated across splits.

### Example (GSM text inference with Llama3 backend)

```bash
python inference_mm_new_model.py \
  --task gsm \
  --backend llama3 \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --lang en \
  --batch_size 16 \
  --max_new_tokens 256 \
  --device cuda:0 \
  --out_file llama3_gsm_predictions.tsv
```

### Example (Geometry3K multimodal inference with MiniCPM backend)

```bash
python inference_mm_new_model.py \
  --task multimodal \
  --backend minicpm \
  --model_path openbmb/MiniCPM-Llama3-V-2_5 \
  --batch_size 4 \
  --max_new_tokens 512 \
  --device cuda:0 \
  --out_file minicpm_geometry3k.parquet
```

## 4. Custom Transformers Components (`transformers_custom`)

The `transformers_custom` package contains lightly modified or extended versions of standard Hugging Face components, such as:

- Model configs: `configuration_minicpm.py`, `configuration_intern_vit.py`, `configuration_internvl_chat.py`
- Model implementations for various families: `modeling_minicpmv.py`, `modeling_idefics.py`, `modeling_llava_next.py`, `modeling_llama.py`, `modeling_qwen2*.py`, `modeling_gemma*.py`, `modeling_intern_vit.py`, `modeling_internvl_chat.py`, etc.
- Tokenization and image processing utilities for MiniCPM-V (`tokenization_minicpmv_fast.py`, `image_processing_minicpmv.py`, `processing_minicpmv.py`, `resampler.py`).
- A local copy of the Hugging Face `generation.py` and `trainer.py` modules, used to ensure compatibility with custom models.

These files are mostly standard or minimally adapted from upstream Transformers and therefore use English comments by default.
You can treat them as drop-in replacements for the corresponding Hugging Face modules when using the models in this submission.
