# Emulating Quantum Dynamics via Curriculum Training

This repository contains the source code for the paper ---. This code can be used to replicate and illustrate the results of the paper.

(Here we will include brief introduction with some images)

## Installation

These instructions are for the Linux Operating System. 

Clone this Repository.

```shell
git clone https://github.com/yaoyu-33/quantum_dynamics_machine_learning
cd quantum_dynamics_machine_learning
```
Create Virtual Environment and Install dependencies
```shell
python -m virtualenv qwave
source qwave/bin/activate
pip install -r requirements.txt
```

Install ffmpeg for rendering simulations
```shell
sudo apt update
sudo apt install ffmpeg
```
(For Windows, make sure the gcc compiler for C is installed in the system)

### Data Preparation

Run the notebook ```data_preparation.ipnyb``` to generate the training data and save it to disk.

### Training

Run the notebook ```model.ipnyb``` to define all the models and train them on the generated data.

### Analysis

Run the notebook ```analyze.ipnyb``` to run and simulate tests on the custom test examples using the trained model.

(Here we can add some more results and plots)


## Citation


## Acknowledgement

