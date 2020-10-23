# VANS
Simulation of noisy is thought to be included. Considering

## history of changes:
### model(circuit) instead of model.predict(circuit)... Maybe we can report this as an issue/bug ?
### We replace the self.expectation_layer with vqe_handler.give_energy, in unitary_killer.
### simplifier remains the same (although the transformation won't be preserved and hence the energy will not remain the same.)

 ## Things that are patched and better solutions are welcome
Rule 5 of utilities.simplifier, we use sympy.solve to reduce many consecutive 1-qubit unitary gates to Rz Rx Rz, but this solution is not very elegant.
<!--
### xxz

[Different configurations and lowest energy found.](https://github.com/matibilkis/vans/blob/genetic/results/xxz/display_results/xxz_4q_20_10.png?raw=true)

[Energies evolution](https://raw.githubusercontent.com/matibilkis/vans/genetic/results/xxz/display_results/plotting_history_energies.png)

[Raw data & ansatz evolution (see /favorite_configuration/evolution.txt)](https://github.com/matibilkis/vans/blob/genetic/results/xxz/)

### TFIM
[Different configurations and lowest energy found](https://github.com/matibilkis/vans/blob/genetic/results/TFIM/tfim4.png?raw_true)

[Energies evolution](https://github.com/matibilkis/vans/blob/genetic/results/TFIM/evolution_energy_TFIM.png?raw=true)

[Raw data & ansatz evolution (see /favorite_configuration/evolution.txt)](https://github.com/matibilkis/vans/blob/genetic/results/TFIM/) -->
