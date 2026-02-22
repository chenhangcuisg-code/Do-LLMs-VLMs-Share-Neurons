# Full pipeline: neuron detection -> compute shared -> merge.
# Paper: Do LLMs and VLMs Share Neurons for Inference? Evidence and Mechanisms of Cross-Modal Transfer

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DETECT_SCRIPT = if ($env:DETECT_SCRIPT) { $env:DETECT_SCRIPT } else { (Join-Path $ScriptDir "detect_mm_add_new.py") }

$NEURON_BASE = if ($env:NEURON_BASE) { $env:NEURON_BASE } else { "./neuron_outputs" }
$OUTPUT_BASE = if ($env:OUTPUT_BASE) { $env:OUTPUT_BASE } else { "./merged_models" }
$CORPUS_PATH = if ($env:CORPUS_PATH) { $env:CORPUS_PATH } else { "." }
$MULTIMODAL_FILE = if ($env:MULTIMODAL_FILE) { $env:MULTIMODAL_FILE } else { "minicpm_vl_math.parquet" }
$SAMPLE_SIZE = if ($env:SAMPLE_SIZE) { $env:SAMPLE_SIZE } else { "100" }
$SKIP_DETECT = if ($env:SKIP_DETECT) { $env:SKIP_DETECT } else { "0" }

New-Item -ItemType Directory -Force -Path $NEURON_BASE | Out-Null
New-Item -ItemType Directory -Force -Path $OUTPUT_BASE | Out-Null

function Run-Detect {
  param($base, $task, $atten, $ffn)
  python $DETECT_SCRIPT `
    --base $base `
    --task $task `
    --atten_ratio $atten `
    --ffn_ratio $ffn `
    --output_path $NEURON_BASE `
    --corpus_path $CORPUS_PATH `
    --multimodal_file $MULTIMODAL_FILE `
    --sample_size $SAMPLE_SIZE `
    --lang en
}

function Compute-Shared {
  param($fileA, $fileB, $outDir)
  python compute_shared_neurons.py $fileA $fileB $outDir
}

# Idefics3
if ($SKIP_DETECT -ne "1") {
  Write-Host "=== Idefics3: detecting neurons (gsm + multimodal) ==="
  Run-Detect "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "gsm" 0.3 0.3
  Run-Detect "HuggingFaceM4/Idefics3-8B-Llama3" "multimodal" 0.3 0.3
  $cmp = "compare_DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3_vs_Idefics3-8B-Llama3_multimodal_en_atten0.3_ffn0.3"
  Compute-Shared "$NEURON_BASE/DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3.json" "$NEURON_BASE/Idefics3-8B-Llama3_multimodal_en_atten0.3_ffn0.3.json" "$NEURON_BASE/$cmp"
}
Write-Host "=== Idefics3: merging ==="
python merge_smart.py --base "HuggingFaceM4/Idefics3-8B-Llama3" --model_a "HuggingFaceM4/Idefics3-8B-Llama3" --model_b "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --json "$NEURON_BASE/compare_DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3_vs_Idefics3-8B-Llama3_multimodal_en_atten0.3_ffn0.3/shared_VL_reasoning.json" --output_dir "$OUTPUT_BASE/idefics3_rank4_beta0_5" --merge_method lora --lora_rank 4 --lora_beta 0.5 --dtype bfloat16 --device auto

# InternVL 3B
if ($SKIP_DETECT -ne "1") {
  Write-Host "=== InternVL 3B: detecting neurons ==="
  Run-Detect "Williammsq/Qwen2.5-3B-Instruct-Math-GRPO" "gsm" 0.3 0.3
  Run-Detect "OpenGVLab/InternVL2_5-4B" "multimodal" 0.3 0.3
  $cmp = "compare_Qwen2.5-3B-Instruct-Math-GRPO_gsm_en_atten0.3_ffn0.3_vs_InternVL2_5-4B_multimodal_en_atten0.3_ffn0.3"
  Compute-Shared "$NEURON_BASE/Qwen2.5-3B-Instruct-Math-GRPO_gsm_en_atten0.3_ffn0.3.json" "$NEURON_BASE/InternVL2_5-4B_multimodal_en_atten0.3_ffn0.3.json" "$NEURON_BASE/$cmp"
}
Write-Host "=== InternVL 3B: merging ==="
python merge_smart.py --base "OpenGVLab/InternVL2_5-4B" --model_a "OpenGVLab/InternVL2_5-4B" --model_b "Williammsq/Qwen2.5-3B-Instruct-Math-GRPO" --json "$NEURON_BASE/compare_Qwen2.5-3B-Instruct-Math-GRPO_gsm_en_atten0.3_ffn0.3_vs_InternVL2_5-4B_multimodal_en_atten0.3_ffn0.3/shared_VL_reasoning.json" --output_dir "$OUTPUT_BASE/internvl_3b_rank32_beta0_5" --merge_method lora --lora_rank 32 --lora_beta 0.5 --dtype float16 --device auto

# LLaVA Next
if ($SKIP_DETECT -ne "1") {
  Write-Host "=== LLaVA Next: detecting neurons ==="
  Run-Detect "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "gsm" 0.3 0.3
  Run-Detect "llava-hf/llama3-llava-next-8b-hf" "multimodal" 0.3 0.3
  $cmp = "compare_DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3_vs_llama3-llava-next-8b-hf_multimodal_en_atten0.3_ffn0.3"
  Compute-Shared "$NEURON_BASE/DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3.json" "$NEURON_BASE/llama3-llava-next-8b-hf_multimodal_en_atten0.3_ffn0.3.json" "$NEURON_BASE/$cmp"
}
Write-Host "=== LLaVA Next: merging ==="
python merge_smart.py --base "llava-hf/llama3-llava-next-8b-hf" --model_a "llava-hf/llama3-llava-next-8b-hf" --model_b "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --json "$NEURON_BASE/compare_DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3_vs_llama3-llava-next-8b-hf_multimodal_en_atten0.3_ffn0.3/shared_VL_reasoning.json" --output_dir "$OUTPUT_BASE/llavanext_rank4_beta0_5" --merge_method lora --lora_rank 4 --lora_beta 0.5 --dtype float16 --device auto

# QwenVL 7B
if ($SKIP_DETECT -ne "1") {
  Write-Host "=== QwenVL 7B: detecting neurons ==="
  Run-Detect "Qwen/Qwen2.5-Math-7B" "gsm" 0.1 0.1
  Run-Detect "Qwen/Qwen2.5-VL-7B-Instruct" "multimodal" 0.1 0.1
  $cmp = "compare_Qwen2.5-Math-7B_gsm_en_atten0.1_ffn0.1_vs_Qwen2.5-VL-7B-Instruct_multimodal_en_atten0.1_ffn0.1"
  Compute-Shared "$NEURON_BASE/Qwen2.5-Math-7B_gsm_en_atten0.1_ffn0.1.json" "$NEURON_BASE/Qwen2.5-VL-7B-Instruct_multimodal_en_atten0.1_ffn0.1.json" "$NEURON_BASE/$cmp"
}
Write-Host "=== QwenVL 7B: merging ==="
python merge_smart.py --base "Qwen/Qwen2.5-VL-7B-Instruct" --model_a "Qwen/Qwen2.5-VL-7B-Instruct" --model_b "Qwen/Qwen2.5-Math-7B" --json "$NEURON_BASE/compare_Qwen2.5-Math-7B_gsm_en_atten0.1_ffn0.1_vs_Qwen2.5-VL-7B-Instruct_multimodal_en_atten0.1_ffn0.1/shared_VL_reasoning.json" --output_dir "$OUTPUT_BASE/qwenvl_7b_rank4_beta0_5" --merge_method lora --lora_rank 4 --lora_beta 0.5 --dtype float16 --device auto

# QwenVL 3B
if ($SKIP_DETECT -ne "1") {
  Write-Host "=== QwenVL 3B: detecting neurons ==="
  Run-Detect "Qwen/Qwen2.5-Math-3B" "gsm" 0.1 0.1
  Run-Detect "Qwen/Qwen2.5-VL-3B-Instruct" "multimodal" 0.1 0.1
  $cmp = "compare_Qwen2.5-Math-3B_gsm_en_atten0.1_ffn0.1_vs_Qwen2.5-VL-3B-Instruct_multimodal_en_atten0.1_ffn0.1"
  Compute-Shared "$NEURON_BASE/Qwen2.5-Math-3B_gsm_en_atten0.1_ffn0.1.json" "$NEURON_BASE/Qwen2.5-VL-3B-Instruct_multimodal_en_atten0.1_ffn0.1.json" "$NEURON_BASE/$cmp"
}
Write-Host "=== QwenVL 3B: merging ==="
python merge_smart.py --base "Qwen/Qwen2.5-VL-3B-Instruct" --model_a "Qwen/Qwen2.5-VL-3B-Instruct" --model_b "Qwen/Qwen2.5-Math-3B" --json "$NEURON_BASE/compare_Qwen2.5-Math-3B_gsm_en_atten0.1_ffn0.1_vs_Qwen2.5-VL-3B-Instruct_multimodal_en_atten0.1_ffn0.1/shared_VL_reasoning.json" --output_dir "$OUTPUT_BASE/qwenvl_3b_rank4_beta0_5" --merge_method lora --lora_rank 4 --lora_beta 0.5 --dtype float16 --device auto

# MiniCPM-V
if ($SKIP_DETECT -ne "1") {
  Write-Host "=== MiniCPM-V: detecting neurons ==="
  Run-Detect "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "gsm" 0.3 0.3
  Run-Detect "openbmb/MiniCPM-Llama3-V-2_5" "multimodal" 0.3 0.3
  $cmp = "compare_DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3_vs_MiniCPM-Llama3-V-2_5_multimodal_en_atten0.3_ffn0.3"
  Compute-Shared "$NEURON_BASE/DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3.json" "$NEURON_BASE/MiniCPM-Llama3-V-2_5_multimodal_en_atten0.3_ffn0.3.json" "$NEURON_BASE/$cmp"
}
Write-Host "=== MiniCPM-V: merging ==="
python merge_smart.py --base "openbmb/MiniCPM-Llama3-V-2_5" --model_a "openbmb/MiniCPM-Llama3-V-2_5" --model_b "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --json "$NEURON_BASE/compare_DeepSeek-R1-Distill-Llama-8B_gsm_en_atten0.3_ffn0.3_vs_MiniCPM-Llama3-V-2_5_multimodal_en_atten0.3_ffn0.3/shared_VL_reasoning.json" --output_dir "$OUTPUT_BASE/minicpm_rank4_beta0_5" --merge_method lora --lora_rank 4 --lora_beta 0.5 --dtype float16 --device auto

Write-Host "=== All experiments completed ==="
