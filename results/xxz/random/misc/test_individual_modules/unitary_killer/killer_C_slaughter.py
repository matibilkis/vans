from unitary_killer import UnitaryMurder
import numpy as np
import cirq
from variational import VQE

killer = UnitaryMurder(n_qubits=3, testing=False)
indices=[]
for k in range(killer.number_of_cnots,killer.number_of_cnots + killer.n_qubits ):
    indices.append(k+killer.n_qubits)
    indices.append(k)
    indices.append(k)


circuit,symbols,index_symbols=killer.give_circuit(indices)

bob = VQE(lr=0.01,epochs=2000)
original_energy, symbols_to_values, _ = bob.vqe(indices)

print("original_energy: ", original_energy)
print(cirq.resolve_parameters(circuit, symbols_to_values))
print(bob.give_energy(indices, symbols_to_values))
print("***")
print(killer.unitary_slaughter(indices, symbols_to_values, index_symbols))
