# Emulator Benchmarks

## Preparation

Download the training and test data

    wget ...  # TODO: specify the right path
    wget ...  # TODO: specify the right path

## Single Training

First, install all dependencies specify in the [requirements.txt](../../requirements.txt)

    pip instll -r ../../requirements.txt

Then, download training and validation data.

    python download_data.py

To train the neural network-based emulator, run

    time PYTHONPATH=./ python train.py &> trainlog.log

You can change parameters of the training
by editing the `config.py` file.

## Slurm

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

    time PYTHONPATH=./ python scalability_benchmark.py -c 4 &> scalabilitylog.log 

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
