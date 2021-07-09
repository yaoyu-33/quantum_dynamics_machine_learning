# Emulator Benchmarks

## Training Data

To get the data, install `gdown`

    pip install gdown

Download the training and test data

    cd ~/
    gdown https://drive.google.com/uc?id=1r1S48SC0NslVOdj89mzVrzcx7082u7Lh

Your md5sum should be

    432f2ee3b3e978efab24e51b6f527dcf  micro_datasets.zip

Unzip the data

    unzip micro_dataset.zip

If you use CUDA, specify the correct path for the XLA, e.g.,

    export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/

## Single Training

First, install all dependencies specify in the [requirements.txt](../../requirements.txt)

    pip instll -r ../../requirements.txt

To train the neural network-based emulator, run

    time PYTHONPATH=./ python train.py --datasets_path=~/micro_dataset/datasets/

You can change parameters of the training
by editing the `global_config.py` file.

## Slurm Job Description

The slurm job description is in the `train.job` file.

To submit a slurm job, run

    sbatch train.job
    
You can check if your job is in the queue by running

    squeue -u $USER

## Scalability Benchmark

Tu run a scalability benchmark we use the [ray library](https://docs.ray.io/en/master/index.html).
First, install ray,

    pip install ray

Then, run

    time PYTHONPATH=./ python scalability_benchmark.py -c 4 --datasets_path=~/micro_dataset/datasets/

Where we specified to use 4 CPUs (see `python scalability_benchmark.py -h` to see all the options).

## Results

TBA

...

## Unit Tests

Install

    pip install pytest

Next, run

    PYTHONPATH=./ py.test tests/

To check the coverage, install

    pip install coverage

Then, run

    coverage run --source='.' -m unittest discover tests "*_test.py"
    coverage report -m --omit=,"tests/*","global_config.py","train.py","scalability_benchmark.py","download_data.py"
