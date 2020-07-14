# with open('vans_gym/solvers/alphabet_w.pickle', 'rb') as handle:
#     b = pickle.load(handle)
#
#
#
# with open('alphabet_w_cirq.pickle', 'wb') as handle:
#     pickle.dump(alphabet, handle, protocol=pickle.HIGHEST_PROTOCOL)



#
#### it would be nice to re-use the model, if it's not necessary to build it again... since it takes a lot of time to build it.

        # W- state alphabet (3 qubits)
        # self.alphabet = {"0": {"gate": cirq.X, "wires": [2]},
        #                  "1": {"gate": cirq.rz, "wires": [0]},
        #                  "2": {"gate": cirq.ry, "wires": [1]},
        #                  "3": {"gate": cirq.CNOT, "wires": [1, 2]},
        #                  "4": {"gate": cirq.CNOT, "wires": [1, 0]},
        #                  "5": {"gate": cirq.ry, "wires": [0]},
        #                  "6": {"gate": cirq.rz, "wires": [0]},
        #                  "7": {"gate": cirq.CNOT, "wires": [0, 1]},
        #                  }
        # with open('alphabet_w.pickle', 'rb') as alphabet:ju
        #     self.alphabet = pickle.load(alphabet)
