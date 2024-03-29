module purge
# module load cray-python
module use /appl/local/csc/modulefiles
module load pytorch

mkdir libs
git clone https://github.com/huggingface/evaluate.git
mv evaluate libs

python -m venv --system-site-packages env
# python -m venv env_2
source env/bin/activate
pip install --upgrade pip setuptools wheel
## pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
## pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
# pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
# pip install transformers
# pip install pytorch-lightning
## pip install lightning-transformers
## pip install torch_optimizer
# export DS_BUILD_FUSED_ADAM=1
# pip install deepspeed
# pip install evaluate
# pip install accelerate
# pip install tensorboard
# pip install seaborn
pip install cer
pip install jiwer
pip install --ignore-installed huggingface-hub
