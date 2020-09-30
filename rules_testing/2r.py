from simplifier import Simplifier
import numpy as np
import cirq
import numpy as np


print("\n  Two consecutive and equal CNOTS compile to identity.")

for nn in [4]:
    print("\n\n\n")
    print("N_QUBITS: ",nn)

    Simp = Simplifier(n_qubits=nn)
    indices=[]


    for j in range(Simp.number_of_cnots + Simp.n_qubits,Simp.number_of_cnots+ 2*Simp.n_qubits):
        indices.append(j)

    for k in range(Simp.number_of_cnots):
        indices.append(k)
        indices.append(k)

    circuit,symbols,index_symbols=Simp.give_circuit(indices)

    print("\n***\n")
    symbols_to_values = {s:k for s,k in zip(symbols, range(len(symbols)))}


    print(Simp.give_unitary(indices,symbols_to_values))
    print(index_symbols)

    Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.simplify_step(indices, symbols_to_values, index_symbols)
    print("\n\n")

    print("ORIGINAL: \n")
    print(Simp.give_unitary(indices,symbols_to_values))
    print("\n")
    print("indices: ", indices)
    print("symbol_to_values: ", symbols_to_values )
    print("index_to_symbols: ", index_symbols)

    print("\nSIMPLIFIED:\n")
    print(Simp.give_unitary(Sindices, Ssymbols_to_values))
    print("\n")
    print("Sindices: ", Sindices)
    print("Ssymbol_to_values", Ssymbols_to_values)
    print("Sindex_to_symbols", Sindex_to_symbols)
