from variational import VQE
from circuit_basics import Basic
import numpy as np
import cirq
from simplifier import Simplifier

print("Here we check that rule 5 + rule 3 preserves the energy. \n")
print("\n Scan for U_3 = Rz Rx Rz, or Rx Rz Rx; if found, abosrb consecutive rz/rx (until a CNOT is found)")
print("The circuit we use is such that the final state, when simplified, is affected, but not the expected value")


NQ=5
Simp = Simplifier(n_qubits=NQ)
indices=[]

for j in range(Simp.number_of_cnots,Simp.number_of_cnots+Simp.n_qubits):
    #indices.append(j+Simp.n_qubits)
    indices.append(j)
    indices.append(j+Simp.n_qubits)
    indices.append(j)
    indices.append(j+Simp.n_qubits)
    indices.append(j)
    indices.append(j)
    indices.append(j+Simp.n_qubits)
    indices.append(j)

#_, d, idx_to_symbols = Simp.give_circuit(indices)
bob = VQE(lr=0.1,epochs=100, n_qubits=NQ)
#print(indices)
#energy1, symbols_to_values, _ = bob.vqe(indices)
#print(energy1)
circuit,symbols,index_symbols=Simp.give_circuit(indices)

print("\n***\n")
symbols_to_values = {s:k for s,k in zip(symbols, range(len(symbols)))}


sindices, ssymbols_to_values, sidx_to_symbols = Simp.reduce_circuit(indices, symbols_to_values,index_symbols )

energy1 = bob.give_energy(indices, symbols_to_values)
energy2 = bob.give_energy(sindices, ssymbols_to_values)

print("ENERGY SHIFT:")
print(energy1-energy2)


print("ORIGINAL:\n")
print(bob.give_unitary(indices,symbols_to_values))

print("SIMPLFIED:\n")
print(bob.give_unitary(sindices,ssymbols_to_values))
