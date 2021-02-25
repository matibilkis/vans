from simplifier import Simplifier
import numpy as np
import cirq
import numpy as np


print("\n  Rule 4: adding values of repeated rotations. It's very easy to do the additions ;)")

for nn in [3]:
    print("\n\n\n")
    print("N_QUBITS: ",nn)

    Simp = Simplifier(n_qubits=nn)
    indices=[9,10,11,6,6,6]


    circuit,symbols,index_symbols=Simp.give_circuit(indices)

    print("\n***\n")
    symbols_to_values = {s:k for s,k in zip(symbols, range(len(symbols)))}


    Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.reduce_circuit(indices, symbols_to_values, index_symbols)
    print("\n\n")

    print("ORIGINAL: \n")
#    print(Simp.give_unitary(indices,symbols_to_values))
    print(Simp.give_circuit(indices)[0])#,symbols_to_values))

#    print(Simp.give_unitary(indices,symbols_to_values))#
#    print(index_symbols,"\n")


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
