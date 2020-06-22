import numpy as np
import pennylane as qml
import pickle


def projector(ket):
    n = len(ket)
    proj = np.zeros((n, n))
    for ind1, i in enumerate(ket):
        for ind2, j in enumerate(ket):
            proj[ind1, ind2] = i*np.conjugate(j)
    return proj


def append_gate(alphabet, index):
    # let's say the parametrized gates are only rotations of 1 free param (how to )
    if "params" in list(alphabet[str(index)].keys()):
        if alphabet[str(index)]["gate"] == qml.Rot:
            params = alphabet[str(index)]["params"]
            return alphabet[str(index)]["gate"](params[0], params[1],params[2], wires=alphabet[str(index)]["wires"])
        else:
            return alphabet[str(index)]["gate"](alphabet[str(index)]["params"][0], wires=alphabet[str(index)]["wires"])
    else:
        return alphabet[str(index)]["gate"](wires=alphabet[str(index)]["wires"])


class PennylaneSolver:
    def __init__(self, n_qubits=3, observable=None):
        self.n_qubits = n_qubits
        self.observable = observable
        self.circuit = None

        self.dev = qml.device("default.qubit", wires=n_qubits)

        with open('alphabet_w.pickle', 'rb') as alphabet:
            self.alphabet = pickle.load(alphabet)

        if observable is None:  # then take projector on W state
            sq = 1 / np.sqrt(3)
            w_state = np.array([0, sq, sq, 0, sq, 0, 0, 0])
            self.observable = qml.Hermitian(projector(w_state), wires=[0, 1, 2])

    def run_circuit(self, list_ops):
        def circuit():
            for op in list_ops:
                append_gate(self.alphabet, int(op))

        @qml.qnode(device=self.dev)
        def circuit_probs():
            circuit()
            return qml.probs(wires=list(range(self.n_qubits)))

        @qml.qnode(device=self.dev)
        def circuit_obs():
            circuit()
            return qml.expval(self.observable)

        energy = circuit_obs()
        probs = circuit_probs()

        return energy, probs


if __name__ == "__main__":
    solver = PennylaneSolver()
    energy, probs = solver.run_circuit([0, 1, 2, 3, 4, 5, 6, 7])

    print(energy, probs)

