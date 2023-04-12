#!/bin/bash
#SBATCH --account=Project_2002820
#SBATCH --time=1:00:00
##SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:4,nvme:64

echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm job nodes: $SLURM_JOB_NODELIST"

module purge
module load pytorch/1.13
export PYTHONUSERBASE=/projappl/project_2000539/rastasii/ecco-ocr-ec/user-env
# pip install --user --ignore-installed pytorch-lightning
# pip install --user --upgrade pip
# pip install --user --upgrade pytorch-lightning
# pip install --user --ignore-installed lightning_utilities
pip install --user evaluate
pip install --user cer
pip install --user jiwer

echo $CUDA_VISIBLE_DEVICES
 
export TMPDIR=$LOCAL_SCRATCH
# export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR
export TRANSFORMERS_CACHE=/scratch/project_2000539/rastasii/transformers_cache
# export OMP_NUM_THREADS=8

# nvcc --version
python -m deepspeed.env_report

NNODES=1
NGPUS=4

TRAIN=$1 # A .jsonl.gz file containing the training samples as {'input', 'output'}.
EVAL=$2 # A .jsonl.gz file containing the evaluation samples, in the same format as the training samples.
OUTDIR=$3 # A directory to which fine-tuning checkpoints are saved.

echo "Copying training data to NVMe."
SECONDS=0
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 gzip -dc $TRAIN > $TMPDIR/train.jsonl
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 gzip -dc $EVAL > $TMPDIR/eval.jsonl
duration=$SECONDS
echo "Train data copied to ${TMPDIR} in ${duration} seconds."
wc -l $TMPDIR/train.jsonl
wc -l $TMPDIR/eval.jsonl

# https://docs.csc.fi/support/tutorials/ml-multi/
if [ $# -eq 3 ]
then
    srun --label python ecco_ocr_ec.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --out_dir $OUTDIR
elif [ $# -eq 4 ]
then
    srun --label python ecco_ocr_ec.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --out_dir $OUTDIR --load_checkpoint $4/$(ls -t1 $4 | grep 'ckpt$' | head -n1)
fi
