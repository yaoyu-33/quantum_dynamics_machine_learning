# Quantum Wave Simulator

## Compile

    gcc -O -o simulator.x simulator.c -lm

## Usage (Example)

    ./simulator.x X0 S0 E0 BH BW EH output_file.txt

Where

 Name | Explanation | Typical Value
 --- | --- | ---
 X0 | the center of the wave packet at time 0 | 40.0
 S0 | the spread of the wave packet at time 0 | 2.0
 E0 | the energy of the wave packet at time 0 | 4.0
 BH | the potential barier hight | 5.0
 BW | the potential barier width | 7.0
 EH | the potential energy at the edges of the system | 0.0
