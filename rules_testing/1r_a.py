from simplifier import Simplifier
import numpy as np
import cirq
import numpy as np


print("\n  Two consecutive and equal CNOTS compile to identity.")

for nn in [3]:
    print("\n\n\n")
    print("N_QUBITS: ",nn)

    Simp = Simplifier(n_qubits=nn)
    indices=[]


    indices = [10,0,6,7,8]

    circuit,symbols,index_symbols=Simp.give_circuit(indices)

    print("\n***\n")
    symbols_to_values = {s:k for s,k in zip(symbols, range(len(symbols)))}


    print(Simp.give_unitary(indices,symbols_to_values))
    print(index_symbols)

    Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.reduce_circuit(indices, symbols_to_values, index_symbols)
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
