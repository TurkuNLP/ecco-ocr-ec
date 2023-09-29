#!/bin/bash
#SBATCH --account=Project_462000241
#SBATCH --time=1:00:00
##SBATCH --time=0:15:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=mi250:1
#SBATCH --exclusive

echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm job nodes: $SLURM_JOB_NODELIST"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export LOCAL_SCRATCH=/tmp

module purge
module load cray-python
# export PYTHONUSERBASE=/projappl/project_2000539/rastasii/ecco-ocr-ec/user-env
# pip install --user --ignore-installed pytorch-lightning
# pip install --user --upgrade pip
# pip install --user --upgrade pytorch-lightning
# pip install --user --ignore-installed lightning_utilities

# cat > $LOCAL_SCRATCH/make_env.sh << "EOF"
# python -m venv $LOCAL_SCRATCH/env
# source $LOCAL_SCRATCH/env/bin/activate
# pip install --upgrade pip setuptools wheel
# # pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
# # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
# pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
# pip install transformers
# pip install pytorch-lightning
# pip install torch_optimizer
# pip install deepspeed
# pip install evaluate
# pip install cer
# pip install jiwer
# pip install tensorboard
# pip install seaborn
# EOF

# srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash $LOCAL_SCRATCH/make_env.sh
# source $LOCAL_SCRATCH/env/bin/activate

source env/bin/activate

echo $CUDA_VISIBLE_DEVICES

export TMPDIR=$LOCAL_SCRATCH
# export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
export HF_HOME=$TMPDIR
# export HF_DATASETS_CACHE=$TMPDIR
export TRANSFORMERS_CACHE=/scratch/project_462000241/rastasii/transformers_cache
# export OMP_NUM_THREADS=8
export HF_EVALUATE_OFFLINE=1

# nvcc --version
python -m deepspeed.env_report
pip show transformers

NNODES=4
NGPUS=8

TRAIN=$1 # A .jsonl.gz file containing the training samples as {'input', 'output'}.
EVAL=$2 # A .jsonl.gz file containing the evaluation samples, in the same format as the training samples.
OUTDIR=$3 # A directory to which fine-tuning checkpoints are saved.

cat > $TMPDIR/copy_script.sh << "EOF"
#!/bin/bash
echo "Temporary directory: $1"
gzip -dc $2 > $1/train.jsonl
gzip -dc $3 > $1/eval.jsonl

# Copy the evaluation metrics for offline use.
cp -r libs/evaluate/metrics/character .
cp -r libs/evaluate/metrics/wer .

echo "Copied training data to $TMPDIR" 
EOF

chmod u+x $TMPDIR/copy_script.sh
sbcast $TMPDIR/copy_script.sh{,}
echo "Copying training data to NVMe."
SECONDS=0
# srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 gzip -dc $TRAIN | sponge $TMPDIR/train.jsonl
# srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 gzip -dc $EVAL | sponge $TMPDIR/eval.jsonl
# srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 ls -la $TMPDIR
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 $TMPDIR/copy_script.sh $TMPDIR $TRAIN $EVAL
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
    srun --label python ecco_ocr_ec.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --out_dir $OUTDIR --load_checkpoint $4
fi
