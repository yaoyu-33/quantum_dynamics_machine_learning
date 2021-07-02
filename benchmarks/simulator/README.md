# Simulator Benchmark

## Preparation

Compile the code

    gcc -O -o simulator.x ../../simulator/quantum_wave_packets/simulator.c -lm

Make sure that you have write access to a temp folder. By default we use the `/tmp` location.

## Benchmark

Run

    time python scalability_benchmark.py -p 4 -n 12

Where the "-p" argument idicates how many processes are created and the "-n" argument indicates how many groups of examples we create.

You can change the parameters of the simulation in the `config.py` file.

## Results

We tested the scalability on a machine with 24 physical and 48 logical cores.

![Simulator Benchmark](../../figures/simulator_benchmark.png "Simulator Scalability Benchmark")

The deviation from the theoretical (ideal) line can be easily understood. We have just 24 physical cores.
The hyperthreading doesn't really help in cpu-heavy operations. Under 24 threads, that we have nearly-ideal scaling.
We also show that the writing/reading operations (at least in the measured regime) are not creating any particular bottleneck.
