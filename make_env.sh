module purge
module load cray-python

python -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
pip install transformers
pip install pytorch-lightning
pip install lightning-transformers
pip install torch_optimizer
pip install deepspeed
pip install evaluate
pip install accelerate
pip install cer
pip install jiwer
pip install tensorboard
pip install seaborn
