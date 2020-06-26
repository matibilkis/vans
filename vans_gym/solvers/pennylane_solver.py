import numpy as np
import pennylane as qml
import pickle


def projector(ket):
    ket = np.expand_dims(ket, 1)
    proj = ket.dot(ket.conjugate().T)
    return proj


class PennylaneSolver:
    def __init__(self, n_qubits=3, observable=None):
        self.n_qubits = n_qubits
        self.observable = observable
        self.circuit = None

        self.dev = qml.device("default.qubit", wires=n_qubits)

        # with open('alphabet_w.pickle', 'rb') as alphabet:
        #     self.alphabet = pickle.load(alphabet)
        self.alphabet = {"0": {"gate": qml.PauliX, "wires": [2]},
                         "1": {"gate": qml.RZ, "wires": [0]},
                         "2": {"gate": qml.RY, "wires": [1]},
                         "3": {"gate": qml.CNOT, "wires": [1, 2]},  #, "params":[np.pi]},
                         "4": {"gate": qml.CNOT, "wires": [1, 0]},  #, "params":[np.pi]},
                         "5": {"gate": qml.RY, "wires": [0]},
                         "6": {"gate":qml.Rot, "wires": [0]},  # borrowed from other optimization
                         "7": {"gate": qml.CNOT, "wires": [0, 1]}  #, "params":[np.pi]},
                   }

        if observable is None:  # then take projector on W state
            sq = 1 / np.sqrt(3)
            w_state = np.array([0, sq, sq, 0, sq, 0, 0, 0])
            self.observable = qml.Hermitian(projector(w_state), wires=[0, 1, 2])

    def build_circuit(self, params, list_ops):
        for i, op in enumerate(list_ops):
            operation = self.alphabet[str(int(op))]
            operation["gate"](*params[i], wires=operation["wires"])

    def run_circuit(self, list_ops):
        @qml.qnode(device=self.dev)
        def circuit_obs(params):
            self.build_circuit(params, list_ops)
            return qml.expval(self.observable)

        @qml.qnode(device=self.dev)
        def circuit_probs(params):
            self.build_circuit(params, list_ops)
            return qml.probs(wires=list(range(self.n_qubits)))

        list_gates = [self.alphabet[str(int(op))]["gate"] for op in list_ops]
        num_params = sum([gate.num_params for gate in list_gates])
        params = [2*np.pi * np.random.sample(gate.num_params) for gate in list_gates]

        # Continuous optimization
        if num_params > 0:
            def loss(x):
                return 1-circuit_obs(x)

            opt = qml.GradientDescentOptimizer(stepsize=0.1)
            steps = 50
            old_loss = loss(params)
            for i in range(steps):
                end = "\r" if i < steps else "\n"
                print(f"Loss: {old_loss}", end=end)
                params = opt.step(loss, params)
                if np.abs(loss(params)-old_loss) < 1e-6:
                    break
                old_loss = loss(params)

        energy, probs = circuit_obs(params), circuit_probs(params)

        return energy, probs


if __name__ == "__main__":
    solver = PennylaneSolver()


    ###### example grid search to show intermediate rewards is not a good idea #####
    # for k,j in zip([6,7],[7,5]):
    #     energy1, _ = solver.run_circuit([0,1,2,3,4,5,4,k])#,j])
    #     energy2, _ = solver.run_circuit([0,1,2,3,4,5,4,k,j])
    #     print("k: {},j: {}, fid1: {}, fid2: {}, sum: {}".format(k,j, energy1, energy2, energy1+energy2))

    # Consider
    # S1(k) = [0,1,2,3,4,5,4,k]
    # S2(k,j) = [0,1,2,3,4,5,4,k,j]
    # k: 6,j: 7, fid_S1: 0.66 fid_S2: 0.99, sum: 1.666
    # k: 7,j: 5, fid_S1: 0.86, fid_S2: 0.866, sum: 1.734
    # where sum is the sum of fidelities between the two time-steps


    # for k in range(5,8):
    #     for j in range(5,8):
    #         energy1, _ = solver.run_circuit([0,1,2,3,4,5,4,k])#,j])
    #         energy2, _ = solver.run_circuit([0,1,2,3,4,5,4,k,j])
    #         print("k: {},j: {}, fid1: {}, fid2: {}, sum: {}".format(k,j, energy1, energy2, energy1+energy2))
    #k: 5,j: 5, fid1: 0.6657161431315839, fid2: 0.665634749011759, sum: 1.3313508921433428
    # k: 5,j: 6, fid1: 0.6661559307247378, fid2: 0.6657556284261439, sum: 1.3319115591508817
    # k: 5,j: 7, fid1: 0.6665047180337567, fid2: 0.9998973747381482, sum: 1.666402092771905
    # k: 6,j: 5, fid1: 0.6654766552638295, fid2: 0.6645147161071819, sum: 1.3299913713710114
    # k: 6,j: 6, fid1: 0.6659358618382097, fid2: 0.6656480041359641, sum: 1.3315838659741739
    # k: 6,j: 7, fid1: 0.6664351178449534, fid2: 0.9919993372477078, sum: 1.658434455092661
    # k: 7,j: 5, fid1: 0.8480531103315252, fid2: 0.8668759416156708, sum: 1.714929051947196
    # k: 7,j: 6, fid1: 0.8617256108352049, fid2: 0.6918543580737462, sum: 1.5535799689089511
    # k: 7,j: 7, fid1: 0.8685549425869972, fid2: 0.666136829654916, sum: 1.534691772241913
    # print(energy, probs)
    # with open('alphabet_w.pickle', 'rb') as alphabet:
    #     aalphabet = pickle.load(alphabet)
    # print(aalphabet)
