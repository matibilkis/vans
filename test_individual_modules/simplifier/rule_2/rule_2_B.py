from simplifier import Simplifier
import numpy as np
import cirq
import numpy as np


print("\n 2. Repeated CNOTs.")



Simp = Simplifier(n_qubits=3)
indices=[]

for j in range(Simp.number_of_cnots + Simp.n_qubits,Simp.number_of_cnots+ 2*Simp.n_qubits):
    indices.append(j)

for k in range(Simp.number_of_cnots):
    indices.append(k)
    indices.append(k)

for j in range(Simp.number_of_cnots ,Simp.number_of_cnots+ Simp.n_qubits):
    indices.append(j)


circuit,symbols,index_symbols=Simp.give_circuit(indices)

print("\n***\n")
symbols_to_values = {s:k for s,k in zip(symbols, range(len(symbols)))}
print(Simp.give_unitary(indices,symbols_to_values))
Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.reduce_circuit(indices, symbols_to_values, index_symbols)
print("\n\n")
print(Simp.give_unitary(Sindices, Ssymbols_to_values))

print("ORIGINAL: \n")
print("indices: ", indices)
print("symbol_to_values: ", symbols_to_values )
print("index_to_symbols: ", index_symbols)

print("\nSIMPLIFIED:\n")
print("Sindices: ", Sindices)
print("Ssymbol_to_values", Ssymbols_to_values)
print("Sindex_to_symbols", Sindex_to_symbols)
