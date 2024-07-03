# create environment. ROSVOT is tested in python3.9, torch 2.1.1, CUDA 11.8
conda create -n rosvot python==3.9
conda activate rosvot
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow==2.9.0 tensorflow-estimator==2.9.0 tensorboardX==2.5
pip install pyyaml matplotlib==3.5 pandas pyworld==0.2.12 librosa torchmetrics
pip install mir_eval pretty_midi pyloudnorm scikit-image textgrid g2p_en npy_append_array einops webrtcvad
export PYTHONPATH=.
