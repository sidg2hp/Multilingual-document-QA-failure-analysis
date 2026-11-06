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
# IIT Roorkee PARAM Ganga template (2x Tesla V100 16GB = 32GB VRAM).
#
# V100 is Volta (sm_70): use FP16 + SDPA. Do NOT enable native bf16 or
# FlashAttention-2.
#
# This is a TEMPLATE, not a verified cluster script. Fill in the module/conda
# activation for your account before submitting. The code does not yet contain
# a confirmatory run; read docs/preregistration.md and docs/peer_review.md first.
# ==============================================================================

set -euo pipefail

mkdir -p outputs/param_ganga outputs/cache

# --- environment: EDIT FOR YOUR ACCOUNT --------------------------------------
# module load python/3.10 cuda/12.1
# source "$HOME/venvs/docqa/bin/activate"
export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"
export TORCH_CUDA_ARCH_LIST="7.0"
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

echo "Starting counterfactual DocQA run on PARAM Ganga"
date
nvidia-smi
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- integration test: English layout control -------------------------------
# Run this first. It exercises the full ladder with gold boxes and catches
# loader/prompt/shape errors before spending GPU time on Indic data.
python scripts/run_experiment.py \
    --config configs/param_ganga_config.yaml \
    --mode funsd \
    --max-samples 100 \
    --resume \
    --seed 42 \
    --output-dir outputs/param_ganga/funsd

# --- Indic transcript-grounded probe ----------------------------------------
# Requires the IndicVisionBench `ocr` split. Questions are generated; the report
# will record the generator version and a warning.
python scripts/run_experiment.py \
    --config configs/param_ganga_config.yaml \
    --mode indic_ocr \
    --max-samples 300 \
    --resume \
    --seed 42 \
    --output-dir outputs/param_ganga/indic_ocr

# --- perturbation sweep on the English control ------------------------------
python scripts/run_experiment.py \
    --config configs/param_ganga_config.yaml \
    --mode funsd \
    --sweep \
    --resume \
    --seed 42 \
    --output-dir outputs/param_ganga/funsd_sweep

# --- analysis (only after the run above is real and reportable) -------------
# python scripts/analyze_results.py \
#     --predictions outputs/param_ganga/funsd/predictions_funsd.jsonl \
#     --manifest manifests/probe_set_funsd.json \
#     --out-dir outputs/param_ganga/analysis

echo "Finished"
date
