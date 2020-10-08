from variational import VQE
from circuit_basics import Basic
import numpy as np
import cirq
from simplifier import Simplifier

print("6. Rz(control) and CNOT(control, target) Rz(control) --> Rz(control) CNOT")

#
#
NQ=3
Simp = Simplifier(n_qubits=NQ)
indices=[9,10,11,0,9,10,11]
#
circuit,symbols,index_symbols=Simp.give_circuit(indices)

bob = VQE(lr=0.1,epochs=100, n_qubits=NQ)
symbols_to_values = {s:k for s,k in zip(symbols, range(len(symbols)))}

energy1 = bob.give_energy(indices, symbols_to_values)

#
#
Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.reduce_circuit(indices, symbols_to_values,index_symbols )
energy2 = bob.give_energy(Sindices, Ssymbols_to_values)
print(energy1 - energy2)

print("ORIGINAL: \n")
print(Simp.give_circuit(indices)[0])#,symbols_to_values))

print("\n")
print("indices: ", indices)
print("symbol_to_values: ", symbols_to_values )
print("index_to_symbols: ", index_symbols)

print("\nSIMPLIFIED:\n")
print(Simp.give_circuit(Sindices)[0])
print("\n")
print("Sindices: ", Sindices)
print("Ssymbol_to_values", Ssymbols_to_values)
print("Sindex_to_symbols", Sindex_to_symbols)
