#!/bin/bash
# Full pipeline: neuron detection -> compute shared -> merge.
# Paper: Do LLMs and VLMs Share Neurons for Inference? Evidence and Mechanisms of Cross-Modal Transfer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECT_SCRIPT="${DETECT_SCRIPT:-${SCRIPT_DIR}/detect_mm_add_new.py}"

# Directories
NEURON_BASE="${NEURON_BASE:-./neuron_outputs}"
OUTPUT_BASE="${OUTPUT_BASE:-./merged_models}"
CORPUS_PATH="${CORPUS_PATH:-.}"
MULTIMODAL_FILE="${MULTIMODAL_FILE:-minicpm_vl_math.parquet}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
# Set to 1 to skip detection (use existing JSON); run only merge
SKIP_DETECT="${SKIP_DETECT:-0}"

mkdir -p "${NEURON_BASE}"
mkdir -p "${OUTPUT_BASE}"

_run_detect() {
  local base="$1"
  local task="$2"
  local atten="$3"
  local ffn="$4"
  python "${DETECT_SCRIPT}" \
    --base "${base}" \
    --task "${task}" \
    --atten_ratio "${atten}" \
    --ffn_ratio "${ffn}" \
    --output_path "${NEURON_BASE}" \
    --corpus_path "${CORPUS_PATH}" \
    --multimodal_file "${MULTIMODAL_FILE}" \
    --sample_size "${SAMPLE_SIZE}" \
    --lang en
}

_compute_shared() {
  local file_a="$1"
  local file_b="$2"
  local out_dir="$3"
  python compute_shared_neurons.py "${file_a}" "${file_b}" "${out_dir}"
}

# =============================================================================
# Idefics3 (atten0.3, ffn0.3, rank=4, beta=0.5)
# =============================================================================
if [ "${SKIP_DETECT}" != "1" ]; then
  echo "=== Idefics3: detecting neurons (gsm + multimodal) ==="
  _run_detect "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "gsm" 0.3 0.3
  _run_detect "HuggingFaceM4/Idefics3-8B-Llama3" "multimodal" 0.3 0.3
  echo "=== Idefics3: computing shared neurons ==="
COMPARE_IDEFICS="compare_DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3_vs_Idefics3-8B-Llama3_multimodal_en_atten0.3_ffn0.3"
  _compute_shared \
    "${NEURON_BASE}/DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3.json" \
    "${NEURON_BASE}/Idefics3-8B-Llama3_multimodal_en_atten0.3_ffn0.3.json" \
    "${NEURON_BASE}/${COMPARE_IDEFICS}"
fi

echo "=== Idefics3: merging ==="
python merge_smart.py \
  --base "HuggingFaceM4/Idefics3-8B-Llama3" \
  --model_a "HuggingFaceM4/Idefics3-8B-Llama3" \
  --model_b "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --json "${NEURON_BASE}/${COMPARE_IDEFICS}/shared_VL_reasoning.json" \
  --output_dir "${OUTPUT_BASE}/idefics3_rank4_beta0_5" \
  --merge_method lora \
  --lora_rank 4 \
  --lora_beta 0.5 \
  --dtype bfloat16 \
  --device auto

# =============================================================================
# InternVL 3B (atten0.3, ffn0.3, rank=32, beta=0.5)
# =============================================================================
if [ "${SKIP_DETECT}" != "1" ]; then
  echo "=== InternVL 3B: detecting neurons (gsm + multimodal) ==="
  _run_detect "Williammsq/Qwen2.5-3B-Instruct-Math-GRPO" "gsm" 0.3 0.3
  _run_detect "OpenGVLab/InternVL2_5-4B" "multimodal" 0.3 0.3
  echo "=== InternVL 3B: computing shared neurons ==="
COMPARE_INTERN="compare_Qwen2.5-3B-Instruct-Math-GRPO_gsm_en_atten0.3_ffn0.3_vs_InternVL2_5-4B_multimodal_en_atten0.3_ffn0.3"
  _compute_shared \
    "${NEURON_BASE}/Qwen2.5-3B-Instruct-Math-GRPO_gsm_en_atten0.3_ffn0.3.json" \
    "${NEURON_BASE}/InternVL2_5-4B_multimodal_en_atten0.3_ffn0.3.json" \
    "${NEURON_BASE}/${COMPARE_INTERN}"
fi

echo "=== InternVL 3B: merging ==="
python merge_smart.py \
  --base "OpenGVLab/InternVL2_5-4B" \
  --model_a "OpenGVLab/InternVL2_5-4B" \
  --model_b "Williammsq/Qwen2.5-3B-Instruct-Math-GRPO" \
  --json "${NEURON_BASE}/${COMPARE_INTERN}/shared_VL_reasoning.json" \
  --output_dir "${OUTPUT_BASE}/internvl_3b_rank32_beta0_5" \
  --merge_method lora \
  --lora_rank 32 \
  --lora_beta 0.5 \
  --dtype float16 \
  --device auto

# =============================================================================
# LLaVA Next (atten0.3, ffn0.3, rank=4, beta=0.5)
# =============================================================================
if [ "${SKIP_DETECT}" != "1" ]; then
  echo "=== LLaVA Next: detecting neurons (gsm + multimodal) ==="
  _run_detect "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "gsm" 0.3 0.3
  _run_detect "llava-hf/llama3-llava-next-8b-hf" "multimodal" 0.3 0.3
  echo "=== LLaVA Next: computing shared neurons ==="
COMPARE_LLAVA="compare_DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3_vs_llama3-llava-next-8b-hf_multimodal_en_atten0.3_ffn0.3"
  _compute_shared \
    "${NEURON_BASE}/DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3.json" \
    "${NEURON_BASE}/llama3-llava-next-8b-hf_multimodal_en_atten0.3_ffn0.3.json" \
    "${NEURON_BASE}/${COMPARE_LLAVA}"
fi

echo "=== LLaVA Next: merging ==="
python merge_smart.py \
  --base "llava-hf/llama3-llava-next-8b-hf" \
  --model_a "llava-hf/llama3-llava-next-8b-hf" \
  --model_b "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --json "${NEURON_BASE}/${COMPARE_LLAVA}/shared_VL_reasoning.json" \
  --output_dir "${OUTPUT_BASE}/llavanext_rank4_beta0_5" \
  --merge_method lora \
  --lora_rank 4 \
  --lora_beta 0.5 \
  --dtype float16 \
  --device auto

# =============================================================================
# QwenVL 7B (atten0.1, ffn0.1, rank=4, beta=0.5)
# =============================================================================
if [ "${SKIP_DETECT}" != "1" ]; then
  echo "=== QwenVL 7B: detecting neurons (gsm + multimodal) ==="
  _run_detect "Qwen/Qwen2.5-Math-7B" "gsm" 0.1 0.1
  _run_detect "Qwen/Qwen2.5-VL-7B-Instruct" "multimodal" 0.1 0.1
  echo "=== QwenVL 7B: computing shared neurons ==="
COMPARE_QWEN7="compare_Qwen2.5-Math-7B_gsm_en_atten0.1_ffn0.1_vs_Qwen2.5-VL-7B-Instruct_multimodal_en_atten0.1_ffn0.1"
  _compute_shared \
    "${NEURON_BASE}/Qwen2.5-Math-7B_gsm_en_atten0.1_ffn0.1.json" \
    "${NEURON_BASE}/Qwen2.5-VL-7B-Instruct_multimodal_en_atten0.1_ffn0.1.json" \
    "${NEURON_BASE}/${COMPARE_QWEN7}"
fi

echo "=== QwenVL 7B: merging ==="
python merge_smart.py \
  --base "Qwen/Qwen2.5-VL-7B-Instruct" \
  --model_a "Qwen/Qwen2.5-VL-7B-Instruct" \
  --model_b "Qwen/Qwen2.5-Math-7B" \
  --json "${NEURON_BASE}/${COMPARE_QWEN7}/shared_VL_reasoning.json" \
  --output_dir "${OUTPUT_BASE}/qwenvl_7b_rank4_beta0_5" \
  --merge_method lora \
  --lora_rank 4 \
  --lora_beta 0.5 \
  --dtype float16 \
  --device auto

# =============================================================================
# QwenVL 3B (atten0.1, ffn0.1, rank=4, beta=0.5)
# =============================================================================
if [ "${SKIP_DETECT}" != "1" ]; then
  echo "=== QwenVL 3B: detecting neurons (gsm + multimodal) ==="
  _run_detect "Qwen/Qwen2.5-Math-3B" "gsm" 0.1 0.1
  _run_detect "Qwen/Qwen2.5-VL-3B-Instruct" "multimodal" 0.1 0.1
  echo "=== QwenVL 3B: computing shared neurons ==="
COMPARE_QWEN3="compare_Qwen2.5-Math-3B_gsm_en_atten0.1_ffn0.1_vs_Qwen2.5-VL-3B-Instruct_multimodal_en_atten0.1_ffn0.1"
  _compute_shared \
    "${NEURON_BASE}/Qwen2.5-Math-3B_gsm_en_atten0.1_ffn0.1.json" \
    "${NEURON_BASE}/Qwen2.5-VL-3B-Instruct_multimodal_en_atten0.1_ffn0.1.json" \
    "${NEURON_BASE}/${COMPARE_QWEN3}"
fi

echo "=== QwenVL 3B: merging ==="
python merge_smart.py \
  --base "Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_a "Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_b "Qwen/Qwen2.5-Math-3B" \
  --json "${NEURON_BASE}/${COMPARE_QWEN3}/shared_VL_reasoning.json" \
  --output_dir "${OUTPUT_BASE}/qwenvl_3b_rank4_beta0_5" \
  --merge_method lora \
  --lora_rank 4 \
  --lora_beta 0.5 \
  --dtype float16 \
  --device auto

# =============================================================================
# MiniCPM-V (atten0.3, ffn0.3, rank=4, beta=0.5)
# =============================================================================
if [ "${SKIP_DETECT}" != "1" ]; then
  echo "=== MiniCPM-V: detecting neurons (gsm + multimodal) ==="
  _run_detect "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "gsm" 0.3 0.3
  _run_detect "openbmb/MiniCPM-Llama3-V-2_5" "multimodal" 0.3 0.3
  echo "=== MiniCPM-V: computing shared neurons ==="
COMPARE_MINICPM="compare_DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3_vs_MiniCPM-Llama3-V-2_5_multimodal_en_atten0.3_ffn0.3"
  _compute_shared \
    "${NEURON_BASE}/DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3.json" \
    "${NEURON_BASE}/MiniCPM-Llama3-V-2_5_multimodal_en_atten0.3_ffn0.3.json" \
    "${NEURON_BASE}/${COMPARE_MINICPM}"
fi

echo "=== MiniCPM-V: merging ==="
python merge_smart.py \
  --base "openbmb/MiniCPM-Llama3-V-2_5" \
  --model_a "openbmb/MiniCPM-Llama3-V-2_5" \
  --model_b "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --json "${NEURON_BASE}/${COMPARE_MINICPM}/shared_VL_reasoning.json" \
  --output_dir "${OUTPUT_BASE}/minicpm_rank4_beta0_5" \
  --merge_method lora \
  --lora_rank 4 \
  --lora_beta 0.5 \
  --dtype float16 \
  --device auto

echo "=== All experiments completed ==="
