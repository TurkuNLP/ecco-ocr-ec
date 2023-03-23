#!/bin/bash
#SBATCH --account=Project_2002820
##SBATCH --time=24:00:00
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=32
#SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:4,nvme:32

module purge
module load pytorch/1.13
# export PYTHONUSERBASE=/projappl/project_2000539/rastasii/ecco-ocr-ec/user-env
# pip install --user --ignore-installed pytorch-lightning
# pip install --user --upgrade pip
# pip install --user --upgrade pytorch-lightning
# pip install --user --ignore-installed lightning_utilities
# echo "PYTHONPATH=" $PYTHONPATH
# export PYTHONPATH=/projappl/project_2000539/rastasii/ecco-ocr-ec/user-env/lib/python3.9/site-packages/
# echo "After exporting, PYTHONPATH=" $PYTHONPATH
# echo "Python version: " $(python --version)
# echo "Python pip version: " $(pip --version)
# export PATH="/projappl/project_2000539/rastasii/ecco-ocr-ec/user-env/bin:$PATH"
# which python
# which python3
# which pip
# pip list

# module load cuda/11.5.0
# module load .unsupported
# module load nvhpc/22.3
# module load tykky
# module list
# nvidia-smi
# nvcc --version
# pip-containerize new --slim --prefix tykky-env req.txt
# export PATH="/projappl/project_2000539/rastasii/ecco-ocr-ec/tykky-env/bin:$PATH"
# printenv > tmp_tykky.txt

echo $CUDA_VISIBLE_DEVICES
 
export TMPDIR=$LOCAL_SCRATCH
# export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR
export TRANSFORMERS_CACHE=/scratch/project_2000539/rastasii/transformers_cache
# export OMP_NUM_THREADS=8

# module load python-data/3.10-22.09
# gcc --version
# nvcc --version
# python -m venv $TMPDIR/env
# source $TMPDIR/env/bin/activate
# pip list
# pip install torch
# pip install transformers
# pip install datasets
# pip install pytorch-lightning
# pip install deepspeed

# tar -zxf env.tar.gz -C $TMPDIR
# source $TMPDIR/env/bin/activate

# python --version
# mkdir $TMPDIR/python-user
# export PYTHONUSERBASE="$TMPDIR/python-user"
# export PATH="$TMPDIR/python-user/bin:$PATH"

# python3 -m pip --version
# python3 -m pip install --user --upgrade setuptools pip wheel
# python3 -m pip --version
# python3 -m pip install --user --upgrade numpy==1.22
# python3 -m pip install --use-pep517 --user nvidia-pyindex
# python3 -m pip install --user nvidia-cuda-runtime-cu12
# python3 -m pip install --user nvidia-cuda-nvcc-cu12
# python3 -m pip install --user --upgrade torch
# python3 -m pip install --user --upgrade transformers
# python3 -m pip install --user --upgrade datasets
# python3 -m pip install --user --upgrade pytorch-lightning
# python3 -m pip install --user --upgrade deepspeed
# python3 -m pip list
# pip install --user --upgrade pytorch-lightning

nvcc --version
python -m deepspeed.env_report

NNODES=1
NGPUS=4

TRAIN=$1
OUTDIR=$2

echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm job nodes: $SLURM_JOB_NODELIST"

echo "Copying training data to NVMe."
SECONDS=0
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 mkdir $TMPDIR/train_files
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 tar -xf $TRAIN -C $TMPDIR/train_files
duration=$SECONDS
echo "Train data copied to ${TMPDIR} in ${duration} seconds."

# https://docs.csc.fi/support/tutorials/ml-multi/
if [ $# -eq 2 ]
then
    srun --label python ecco_ocr_ec.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train_files --out_dir $OUTDIR
elif [ $# -eq 3 ]
then
    srun --label python ecco_ocr_ec.py --nodes $NNODES --gpus $NGPUS --train $TMPDIR/train_files --out_dir $OUTDIR --load_checkpoint $(ls -t $2/*.ckpt | head -n1)
fi
