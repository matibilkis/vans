from simplifier import Simplifier
import numpy as np
import cirq
import numpy as np


print("\n Rule 3 - Rotation around z axis of |0> only adds phase hence leaves invariant <H>.")

print("Case 1: ran out of gates.-")
indices = [6,7,8]#,#9,10,11]
Simp = Simplifier(n_qubits=3)
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
print("Case 2: ansatz complex enough")

indices = [6,7,8,9,10,11,6,7,8]
Simp = Simplifier(n_qubits=3)
circuit,symbols,index_symbols=Simp.give_circuit(indices)

symbols_to_values = {s:k for s,k in zip(symbols, range(len(symbols)))}
print(Simp.give_unitary(indices,symbols_to_values))
Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.simplify_step(indices, symbols_to_values, index_symbols)
print("\n\n")
print(Simp.give_unitary(Sindices, Ssymbols_to_values))
print("\n")

print("ORIGINAL: \n")
print("indices: ", indices)
print("symbol_to_values: ", symbols_to_values )
print("index_to_symbols: ", index_symbols)

print("\nSIMPLIFIED:\n")
print("Sindices: ", Sindices)
print("Ssymbol_to_values", Ssymbols_to_values)
print("Sindex_to_symbols", Sindex_to_symbols)
