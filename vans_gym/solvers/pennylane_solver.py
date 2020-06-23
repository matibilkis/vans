import numpy as np
import pennylane as qml
import pickle


def projector(ket):
    ket = np.expand_dims(ket, 1)
    proj = ket.dot(ket.conjugate().T)
    return proj

#
# def append_gate(alphabet, index):
#     # let's say the parametrized gates are only rotations of 1 free param (how to )
#     if "params" in list(alphabet[str(index)].keys()):
#         if alphabet[str(index)]["gate"] == qml.Rot:
#             params = alphabet[str(index)]["params"]
#             return alphabet[str(index)]["gate"](params[0], params[1], params[2], wires=alphabet[str(index)]["wires"])
#         else:
#             return alphabet[str(index)]["gate"](alphabet[str(index)]["params"][0], wires=alphabet[str(index)]["wires"])
#     else:
#         return alphabet[str(index)]["gate"](wires=alphabet[str(index)]["wires"])




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

    def return_list_pars(self,list_ops):
        #list_ops specifies the gates to append. This function is to be able to compute the gradient.
        #The gates we use are qml.Rot (3 free parameters), or RX, RY, RZ (only one free param).
        #optimal_route for W-state = [0,1,2,3,4,5,4,6,7]
        ind_par=0
        p=[]
        for k in list_ops:
            operation = self.alphabet[str(int(k))]
            if "params" in operation.keys():
                if operation["gate"].num_params == 1:
                    p.append(qml.variable.Variable(idx=ind_par))
                    ind_par+=1
                else:
                    for ind1 in range(3):
                        p.append(qml.variable.Variable(idx=ind_par+ind1))
                    ind_par+=3
        return p



    def run_circuit(self, list_ops):
        @qml.qnode(device=self.dev)
        def circuit_obs(params, list_ops):
            ind_par=0
            for k in list_ops:
                operation = self.alphabet[str(int(k.val))]
                if "params" in operation.keys():
                    if operation["gate"].num_params == 1:
                        operation["gate"](params[ind_par], wires = operation["wires"])
                        ind_par+=1
                    else:
                        operation["gate"](params[ind_par],params[ind_par+1],params[ind_par+2], wires = operation["wires"])
                        ind_par+=3
                else:
                    operation["gate"](wires = operation["wires"])
            return qml.expval(self.observable)

        @qml.qnode(device=self.dev)
        def circuit_probs(params, list_ops):
            ind_par=0
            for k in list_ops:
                operation = self.alphabet[str(int(k.val))]
                if "params" in operation.keys():
                    if operation["gate"].num_params == 1:
                        operation["gate"](params[ind_par], wires = operation["wires"])
                        ind_par+=1
                    else:
                        operation["gate"](params[ind_par],params[ind_par+1],params[ind_par+2], wires = operation["wires"])
                        ind_par+=3
                else:
                    operation["gate"](wires = operation["wires"])

            return qml.probs(wires=list(range(self.n_qubits)))


        def optimize_continuous(list_ops):
            #
            pars = self.return_list_pars(list_ops)

            if len(pars)==0:
                return circuit_obs([],list_ops), circuit_probs([],list_ops)
            else:

                def loss(x):
                    return 1-circuit_obs(x, list_ops)

                opt = qml.GradientDescentOptimizer(stepsize=0.1)
                steps = 100
                params = np.random.sample(len(pars))
                for i in range(steps):
                    params = opt.step(loss, params)
                return circuit_obs(params,list_ops), circuit_probs(params,list_ops)

        energy, probs = optimize_continuous(list_ops)

        return energy, probs


if __name__ == "__main__":
    solver = PennylaneSolver()
    # energy, probs = solver.run_circuit([0,1,2,3,4,5,4,6,7])
    # energy, probs = solver.run_circuit([0])#,1,2,3,4,5,4,6,7])

    print(energy, probs)
