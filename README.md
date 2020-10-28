# VANS
noiseless

## Things that are patched and better solutions are welcome
Rule 5 of utilities.simplifier, we use sympy.solve to reduce many consecutive 1-qubit unitary gates to Rz Rx Rz, but this solution is not very elegant.

### xxz

[Different configurations and lowest energy found.](https://github.com/matibilkis/vans/blob/genetic/results/xxz/display_results/xxz_4q_20_10.png?raw=true)

[Energies evolution](https://raw.githubusercontent.com/matibilkis/vans/genetic/results/xxz/display_results/plotting_history_energies.png)

[Raw data & ansatz evolution (see /favorite_configuration/evolution.txt)](https://github.com/matibilkis/vans/blob/genetic/results/xxz/)

### TFIM
[Different configurations and lowest energy found](https://github.com/matibilkis/vans/blob/genetic/results/TFIM/tfim4.png?raw_true)

[Energies evolution](https://github.com/matibilkis/vans/blob/genetic/results/TFIM/evolution_energy_TFIM.png?raw=true)

[Raw data & ansatz evolution (see /favorite_configuration/evolution.txt)](https://github.com/matibilkis/vans/blob/genetic/results/TFIM/)

## Noisy circuits
We implement noisy channels as a weighted average of unitary transformations. This allows to use the fast C++ TFQ simulator, since DensityMatrixSimulator is not implemented (yet). Quite arbitrarly, the channel acts before each gate appears in the circuit (in case of CNOT, the channel is encountered at both control and target). For example

Running VANS in this context is feasible, but sligthly expensive (at least on my laptop, without GPU and depending on the particular circuit, optimization takes more than 10 minutes and hence that VANS-iteration step is considered skipped). From a particular 50-iterations-VANS run - and three different values of depolarizing channel -, we observe a nice reduction of circuit's number of CNOTS for only one case, but the remaining two got stuck. Find the circuits generating the results [here](https://github.com/matibilkis/vans/blob/implicit_noise/noisy_TFIM_3qubits):

![depo](https://github.com/matibilkis/vans/blob/implicit_noise/noisy_TFIM_3qubits/depolarizing_tfim_3qubits.png)
