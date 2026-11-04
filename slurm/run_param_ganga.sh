#!/bin/bash
#SBATCH --job-name=docqa_error_attr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_%j.err

# ==============================================================================
# IIT Roorkee PARAM Ganga Batch Script (2x NVIDIA Tesla V100 16GB = 32GB VRAM)
# Note: V100 is Volta (sm_70) -> Use FP16 + SDPA (do NOT enable bf16 or FA2)
# ==============================================================================

set -euo pipefail

mkdir -p outputs/param_ganga

export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"
export TORCH_CUDA_ARCH_LIST="7.0"
export CUDA_VISIBLE_DEVICES=0,1

echo "Starting Counterfactual DocQA Attribution & Routing Run on PARAM Ganga..."
nvidia-smi

python scripts/run_experiment.py \
    --config configs/param_ganga_config.yaml \
    --mode indic_vision_bench \
    --max-samples 500
