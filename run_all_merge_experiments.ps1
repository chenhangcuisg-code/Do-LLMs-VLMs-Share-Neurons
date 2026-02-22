# Run all merge experiments with fixed parameters per model type.
# Paper: Do LLMs and VLMs Share Neurons for Inference? Evidence and Mechanisms of Cross-Modal Transfer

$ErrorActionPreference = "Stop"

$JSON_BASE = if ($env:JSON_BASE) { $env:JSON_BASE } else { "./neuron_outputs" }
$OUTPUT_BASE = if ($env:OUTPUT_BASE) { $env:OUTPUT_BASE } else { "./merged_models" }

Write-Host "=== Idefics3 (rank=4, beta=0.5) ==="
python merge_smart.py `
  --base "HuggingFaceM4/Idefics3-8B-Llama3" `
  --model_a "HuggingFaceM4/Idefics3-8B-Llama3" `
  --model_b "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" `
  --json "$JSON_BASE/compare_DeepSeek_R1_Distill_Llama_8B_gsm_en_atten0_3_ffn0_3_vs_Idefics3_8B_Llama3_multimodal_en_atten0_3_ffn0_3/shared_VL_reasoning.json" `
  --output_dir "$OUTPUT_BASE/idefics3_rank4_beta0_5" `
  --merge_method lora `
  --lora_rank 4 `
  --lora_beta 0.5 `
  --dtype bfloat16 `
  --device auto

Write-Host "=== InternVL 3B (rank=32, beta=0.5) ==="
python merge_smart.py `
  --base "OpenGVLab/InternVL2_5-4B" `
  --model_a "OpenGVLab/InternVL2_5-4B" `
  --model_b "Williammsq/Qwen2.5-3B-Instruct-Math-GRPO" `
  --json "$JSON_BASE/3b_wil_intern/shared_VL_reasoning.json" `
  --output_dir "$OUTPUT_BASE/internvl_3b_rank32_beta0_5" `
  --merge_method lora `
  --lora_rank 32 `
  --lora_beta 0.5 `
  --dtype float16 `
  --device auto

Write-Host "=== LLaVA Next (rank=4, beta=0.5) ==="
python merge_smart.py `
  --base "llava-hf/llama3-llava-next-8b-hf" `
  --model_a "llava-hf/llama3-llava-next-8b-hf" `
  --model_b "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" `
  --json "$JSON_BASE/llavanext/shared_VL_reasoning.json" `
  --output_dir "$OUTPUT_BASE/llavanext_rank4_beta0_5" `
  --merge_method lora `
  --lora_rank 4 `
  --lora_beta 0.5 `
  --dtype float16 `
  --device auto

Write-Host "=== QwenVL 7B (rank=4, beta=0.5) ==="
python merge_smart.py `
  --base "Qwen/Qwen2.5-VL-7B-Instruct" `
  --model_a "Qwen/Qwen2.5-VL-7B-Instruct" `
  --model_b "Qwen/Qwen2.5-Math-7B" `
  --json "$JSON_BASE/qwenvl_7b/shared_VL_reasoning.json" `
  --output_dir "$OUTPUT_BASE/qwenvl_7b_rank4_beta0_5" `
  --merge_method lora `
  --lora_rank 4 `
  --lora_beta 0.5 `
  --dtype float16 `
  --device auto

Write-Host "=== QwenVL 3B (rank=4, beta=0.5) ==="
python merge_smart.py `
  --base "Qwen/Qwen2.5-VL-3B-Instruct" `
  --model_a "Qwen/Qwen2.5-VL-3B-Instruct" `
  --model_b "Qwen/Qwen2.5-Math-3B" `
  --json "$JSON_BASE/qwenvl_3b/shared_VL_reasoning.json" `
  --output_dir "$OUTPUT_BASE/qwenvl_3b_rank4_beta0_5" `
  --merge_method lora `
  --lora_rank 4 `
  --lora_beta 0.5 `
  --dtype float16 `
  --device auto

Write-Host "=== MiniCPM-V (rank=4, beta=0.5) ==="
python merge_smart.py `
  --base "openbmb/MiniCPM-Llama3-V-2_5" `
  --model_a "openbmb/MiniCPM-Llama3-V-2_5" `
  --model_b "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" `
  --json "$JSON_BASE/minicpm/shared_VL_reasoning.json" `
  --output_dir "$OUTPUT_BASE/minicpm_rank4_beta0_5" `
  --merge_method lora `
  --lora_rank 4 `
  --lora_beta 0.5 `
  --dtype float16 `
  --device auto

Write-Host "=== All merge experiments completed ==="
