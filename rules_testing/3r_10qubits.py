from simplifier import Simplifier
import numpy as np
import cirq
import numpy as np


print("\n Rule 3 - Rotation around z axis of |0> only adds phase hence leaves invariant <H>.")

print("Case 1: ran out of gates.-")
Simp = Simplifier(n_qubits=10)

indices = np.arange(Simp.number_of_cnots,Simp.n_qubits+Simp.number_of_cnots)
circuit,symbols,index_symbols=Simp.give_circuit(indices)

print("\n***\n")
symbols_to_values = {s:k for s,k in zip(symbols, range(len(symbols)))}
print(Simp.give_unitary(indices,symbols_to_values))
Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.simplify_step(indices, symbols_to_values, index_symbols)
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

print("\n\n\n*****")
print("*****\n\n\n")


for k in range(3):
    print("\n")

print("CASE 2: ansatz complex enough")
Simp = Simplifier(n_qubits=10)

indices = list(range(Simp.number_of_cnots,Simp.n_qubits+Simp.number_of_cnots))
for k in range(Simp.number_of_cnots):
    indices.append(k)
for k in list(range(Simp.number_of_cnots+Simp.n_qubits,2*Simp.n_qubits+Simp.number_of_cnots)):
    indices.append(k)

circuit,symbols,index_symbols=Simp.give_circuit(indices)

print("\n***\n")
symbols_to_values = {s:k for s,k in zip(symbols, range(len(symbols)))}
print(Simp.give_unitary(indices,symbols_to_values))
Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.simplify_step(indices, symbols_to_values, index_symbols)
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

print("\n\n\n*****")
print("*****\n\n\n")
