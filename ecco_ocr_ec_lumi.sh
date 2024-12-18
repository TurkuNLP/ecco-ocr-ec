#!/bin/bash
#SBATCH --account=Project_462000587
##SBATCH --time=12:00:00
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --gpus-per-task=mi250:8
#SBATCH --exclusive

echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm job nodes: $SLURM_JOB_NODELIST"

export NCCL_SOCKET_IFNAME=hsn
export LOCAL_SCRATCH=/tmp

module purge
# module load LUMI partition/container EasyBuild-user
# module load cray-python
# eb PyTorch-2.1.0-rocm-5.6.1-python-3.10-singularity-20231123.eb
# module load PyTorch/2.1.0-rocm-5.6.1-python-3.10-singularity-20231123
module use /appl/local/csc/modulefiles
module load pytorch
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

# rm -r env_2
# bash make_env.sh
source env/bin/activate

# echo "SIF: $SIF"
# echo "RUNSCRIPTS: $RUNSCRIPTS"
# ls $RUNSCRIPTS

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HIP_PATH: $HIP_PATH"
echo "HSA_PATH: $HSA_PATH"
echo "HIP_ROCCLR_HOME: $HIP_ROCCLR_HOME"
echo "HIP_CLANG_PATH: $HIP_CLANG_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

export TMPDIR=$LOCAL_SCRATCH
# export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
export TORCH_HOME=/scratch/project_462000587/rastasii/cache
export HF_HOME=/scratch/project_462000587/rastasii/cache
# export HF_DATASETS_CACHE=$TMPDIR
# export TRANSFORMERS_CACHE=/scratch/project_462000347/rastasii/cache
# export OMP_NUM_THREADS=8
export HF_EVALUATE_OFFLINE=1

# nvcc --version
# rocminfo
which python
python -m deepspeed.env_report
pip show transformers
pip show accelerate

NNODES=1
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
    # srun --label singularity exec -B "/appl:/appl" -B "$SCRATCH:$LOCAL_SCRATCH" $SIF $RUNSCRIPTS/conda-python-simple ecco_ocr_ec.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --out_dir $OUTDIR
    srun --label python ecco_ocr_ec.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --out_dir $OUTDIR
elif [ $# -eq 4 ]
then
    # python $4/zero_to_fp32.py $4 $4/pytorch_model.bin
    # srun --label python ecco_ocr_ec.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --out_dir $OUTDIR --load_checkpoint $4
    # scontrol show hostnames "$SLURM_JOB_NODELIST"
    # export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    # python -m accelerate.commands.launch --num_processes $(($NNODES * $NGPUS)) ecco_ocr_ec_eval.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --out_dir $OUTDIR --load_checkpoint $4
    # python -m accelerate.commands.launch --num_processes $(($NNODES * $NGPUS)) ocr_ec_inference.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --predictions $TMPDIR/predictions.jsonl --references $TMPDIR/references.jsonl --load_checkpoint $4
    # singularity_wrapper exec deepspeed ocr_ec_inference.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --predictions $TMPDIR/predictions.jsonl --references $TMPDIR/references.jsonl --load_checkpoint $4
    # python extract_ckpt_from_pl.py --ds_checkpoint $4 --model_path $4
    srun --label python ocr_ec_inference.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train.jsonl --eval $TMPDIR/eval.jsonl --predictions $TMPDIR/predictions.jsonl --references $TMPDIR/references.jsonl --load_checkpoint $4
    python ../ocr-postcorrection-lm/evaluation/eval_metrics.py -p $TMPDIR/predictions.jsonl -r $TMPDIR/references.jsonl -m all
fi
