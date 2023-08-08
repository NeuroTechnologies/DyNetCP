#conda create -n dsap_journal python=3.7
#conda activate dsap_journal

conda install numpy==1.21.2 matplotlib==2.2.3 scipy==1.7.3 pandas==1.3.4 h5py==2.10.0 markupsafe==2.0.1 pyyaml==5.3.1 tensorboard==2.2.1
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install allensdk==2.2.0

pip install -e ./