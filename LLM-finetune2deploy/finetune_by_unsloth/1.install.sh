conda create --name unslot python==3.10 -y
conda activate unsloth
conda install nvidia/label/cuda-12.1.0::cuda -y
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia


pip install unsloth
#pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"