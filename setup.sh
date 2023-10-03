conda create -n torch-gpu python=3.8 pip zip unzip gh -c conda-forge
conda activate torch-gpu

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers ipykernel click sentencepiece