## Installation

This code was originally run using Python 3.7 on a Ubuntu 16.04.2 machine with a GeForce GTX 1080 GPU. It requires installing the following libraries:
- [PyTorch](https://pytorch.org/get-started/previous-versions/) (1.7.1)
- numpy (1.21.2)
- matplotlib (2.2.3)
- allensdk (2.2.0)
- pandas (1.3.4)
- h5py (2.10.0)
- pyyaml (5.3.1)
- tensorboard (2.2.1)
- tqdm

It is recommended to use an Anaconda environment to install everything.
To finish code setup, run the following command from the root directory:

`pip install -e ./`

Installation should take less than 10 minutes.

## Running
To train the model for a single session of the Allen Brain Institute data, run the following script:
`./scripts/run_on_session.sh`

To train the model on the synthetic Hodgkin-Huxley spiking data, you must first download the files from [this link](https://github.com/NII-Kobayashi/GLMCC/tree/master/simulation_data) and place them in the directory `processed_data/hh_synth/`. Then, run the following script:
`./scripts/run_on_hh.sh`

To train the model on our synthetic data, run the following script:
`./scripts/run_on_synth.sh`

## License
This code is released under the MIT License.