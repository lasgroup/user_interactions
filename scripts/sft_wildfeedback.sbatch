#!/bin/bash
#SBATCH --job-name=qwen3-sft
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240000
#SBATCH --time=04:00:00
#SBATCH --output=/users/%u/output/wildfeedback-sft/%j.log
#SBATCH --error=/users/%u/output/wildfeedback-sft/%j.err
#SBATCH --environment=sample-efficient-rlvr
#SBATCH --exclude=nid007004,nid006606,nid006684,nid006638,nid006569,nid006622,nid006571,nid006877,nid006817,nid006866,nid006927,nid007223

REPO="/users/$USER/test-time-alignment"
CONFIG_PATH="$REPO/multigpu_accelerate_config.yaml"

SCRATCH_DIR="${SCRATCH:-/tmp}/qwen3-sft-${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"/{hf,datasets,hub,wandb,pip}

export HF_HOME="$SCRATCH_DIR/hf"
export HF_DATASETS_CACHE="$SCRATCH_DIR/datasets"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/hub"
export WANDB_DIR="$SCRATCH_DIR/wandb"
export PIP_CACHE_DIR="$SCRATCH_DIR/pip"
export TOKENIZERS_PARALLELISM="false"
export WANDB_PROJECT="wildfeedback-sft"
export WANDB_NAME="qwen3-4b-sft-${SLURM_JOB_ID}"

# If you need auth, set HF_TOKEN in your environment (don’t hardcode it here)
# export HF_TOKEN="..."

export WANDB_PROJECT="wildfeedback-sft"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

cd "$REPO" || exit 1

python -m pip install --user --upgrade pip

python -m pip install --user -U transformers trl accelerate datasets wandb bitsandbytes

export OUTPUT_DIR="/capstor/scratch/cscs/$USER/wildfeedback-sft/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

export SLURM_CPU_BIND=quiet,none

# Choose one:
# MODEL_NAME="Qwen/Qwen3-4B-Base"
MODEL_NAME="Qwen/Qwen3-4B"

echo "Using model: $MODEL_NAME"

DATA_PATH="$REPO/data/wildfeedback_interactions_v2.jsonl"

echo "Starting SFT..."
echo "Model:  $MODEL_NAME"
echo "Data:   $DATA_PATH"
echo "Out:    $OUTPUT_DIR"
echo "Accel:  $CONFIG_PATH"

srun --ntasks=1 --cpu-bind=none bash -lc "
  accelerate launch --config_file \"$CONFIG_PATH\" \"$REPO/sft_wildfeedback.py\" \
    --model \"$MODEL_NAME\" \
    --data \"$DATA_PATH\" \
    --out \"$OUTPUT_DIR\" \
    --max_length 2048 \
    --epochs 2 \
    --lr 2e-6 \
    --bsz 4 \
    --gas 2
"
