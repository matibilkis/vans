import numpy as np
import sympy
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

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


class CirqSolver:
    def __init__(self, n_qubits=3, observable_name=None, *observable_from_matrix):
        self.name = "CirqSolver"
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.observable_name = observable_name

        self.alphabet = {"0": {"gate": cirq.X, "wires": [0]},
                         "1": {"gate": cirq.X, "wires": [1]},
                         "2": {"gate": cirq.X, "wires": [2]},
                         "3": {"gate": cirq.H, "wires": [0]},
                         "4": {"gate": cirq.H, "wires": [1]},
                         "5": {"gate": cirq.H, "wires": [2]},
                        }

        self.parametrized = [cirq.rz, cirq.ry, cirq.rx]
        self.target_reward = self.n_qubits #mostly for the ising high transv fields.

        if self.observable_name is "W-state":  # then take projector on W state
            sq = 1 / np.sqrt(3)
            w_state = np.array([0, sq, sq, 0, sq, 0, 0, 0])
            w_proj = cirq.density_matrix_from_state_vector(w_state)
            self.observable = self.cirq_friendly_observable(w_proj)
            self.observable_matrix = w_proj
            self.target_reward = 1

        elif self.observable_name == "Ising_High_TFields_HX":
            self.observable = [cirq.X.on(q) for q in self.qubits] # -J \sum_{i} Z_i Z_{i+1} - g \sum_i X_i    when g>>J
            self.observable_matrix = cirq.unitary(cirq.Circuit(self.observable))
            self.alphabet = {}
            for k in range(self.n_qubits):
                self.alphabet[str(k)] = {"gate": cirq.X, "wires":[k]}
                self.alphabet[str(k+self.n_qubits)] = {"gate": cirq.H, "wires":[k]}

        elif self.observable_name == "Ising_High_TFields_rots":
            self.observable = [cirq.X.on(q) for q in self.qubits] # \sum X_i
            self.observable_matrix = cirq.unitary(cirq.Circuit(self.observable))
            self.alphabet = {}
            for k in range(self.n_qubits):
                self.alphabet[str(k)] = {"gate": cirq.ry, "wires":[k]}

        elif self.observable_name == "Ising_High_TFields_hybrid_2":
            self.observable = [cirq.X.on(q) for q in self.qubits] # \sum X_i
            self.observable_matrix = cirq.unitary(cirq.Circuit(self.observable))
            self.alphabet = {"0": {"gate": cirq.Z, "wires": [0]},
                             "1": {"gate": cirq.Z, "wires": [1]},
                             "2": {"gate": cirq.H, "wires": [0]},
                             "4": {"gate": cirq.ry, "wires": [1]},
                            }

        elif self.observable_name == "Ising_High_TFields_hybrid_3":
            self.observable = [cirq.X.on(q) for q in self.qubits] # \sum X_i
            self.observable_matrix = cirq.unitary(cirq.Circuit(self.observable))
            self.alphabet = {"0": {"gate": cirq.Z, "wires": [0]},
                             "1": {"gate": cirq.Z, "wires": [1]},
                             "2": {"gate": cirq.Z, "wires": [2]},
                             "3": {"gate": cirq.H, "wires": [0]},
                             "4": {"gate": cirq.rx, "wires": [1]},
                             "5": {"gate": cirq.ry, "wires": [2]},
                            }
        else:
            print("check which observable you are voptimizing!")
            self.observable = observable_from_matrix
            self.observable_matrix = cirq.unitary(cirq.Circuit(self.observable))
            self.target_reward = 100 #just to say smthg

    def cirq_friendly_observable(self, obs):
        PAULI_BASIS = {
            'I': np.eye(2),
            'X': np.array([[0., 1.], [1., 0.]]),
            'Y': np.array([[0., -1j], [1j, 0.]]),
            'Z': np.diag([1., -1]),
        }

        pauli3 = cirq.linalg.operator_spaces.kron_bases(PAULI_BASIS, repeat=3)
        decomp = cirq.linalg.operator_spaces.expand_matrix_in_orthogonal_basis(obs, pauli3)

        PAULI_BASIS_CIRQ = {
            'I': cirq.X,
            'X': cirq.X,
            'Y': cirq.Y,
            'Z': cirq.Z,
        }

        unt = []
        for term in decomp.items():
            gate_name = term[0]
            coeff = term[1]
            s = 0
            ot = float(coeff)
            for qpos, single_gate in enumerate(gate_name):
                if single_gate == "I":
                    ot *= PAULI_BASIS_CIRQ[single_gate](self.qubits[qpos])*PAULI_BASIS_CIRQ[single_gate](self.qubits[qpos])
                else:
                    ot *= PAULI_BASIS_CIRQ[single_gate](self.qubits[qpos])
            if s < 3:
                unt.append(ot)
        return unt

    def vansatz_keras_model(self, vansatz, observable):
        # notice observable may in general be expressed as linear combination
        # of different elements  on orthonormal basis obtained from tensor product
        # of SU(2) generators. tf.math.reduce_sum is in charge of taking this linear combination.
        circuit_input = tf.keras.Input(shape=(), dtype=tf.string)
        output = tfq.layers.Expectation()(
                circuit_input,
                symbol_names=vansatz.symbols,
                operators=tfq.convert_to_tensor([observable]),
                initializer=tf.keras.initializers.RandomNormal()) #notice this is not strictly necessary.

        output = tf.math.reduce_sum(output, axis=-1, keepdims=True)
        model = tf.keras.Model(inputs=circuit_input, outputs=output)
        adam = tf.keras.optimizers.Adam(learning_rate=0.1)
        model.compile(optimizer=adam, loss='mse')
        return model

    def run_circuit(self, list_ops):
        wst = VAnsatz(self.n_qubits,
                      self.observable_name,
                      self.target_reward,
                      list_ops)

        if len(wst.symbols) == 0:
            if self.observable_name == "W-state" or self.observable_name is None:
                simulator = cirq.Simulator()
                result = simulator.simulate(wst.get_state(self.qubits, params=np.random.sample(len(wst.symbols))), qubit_order=self.qubits)
                energy = np.trace(np.dot(wst.observable_matrix, cirq.density_matrix_from_state_vector(result.final_state))).real
                probs = np.abs(result.final_state)**2
                return energy, probs
            else:
                ci = tfq.convert_to_tensor([wst.circuit])
                expval = tfq.layers.Expectation()(
                                                ci,
                                                operators=tfq.convert_to_tensor([self.observable]))
                energy = np.squeeze(tf.math.reduce_sum(expval, axis=-1, keepdims=True))
                simulator = cirq.Simulator()
                result = simulator.simulate(wst.get_state(self.qubits), qubit_order=self.qubits)
                probs = np.abs(result.final_state)**2
                return energy, probs

        model = self.vansatz_keras_model(wst, self.observable)
        w_input = tfq.convert_to_tensor([wst.circuit])
        w_output = tf.ones((1, 1))*self.target_reward
        model.fit(x=w_input, y=w_output, batch_size=1, epochs=50, verbose=0)
        energy = float(np.squeeze(model.predict(w_input)))

        simulator = cirq.Simulator()
        result = simulator.simulate(wst.get_state(self.qubits,params=model.get_weights()[0]), qubit_order=self.qubits)
        probs = np.abs(result.final_state)**2
        return energy, probs



class VAnsatz(CirqSolver):
    def __init__(self, n_qubits, observable_name, target_reward, trajectory):
        super(VAnsatz, self).__init__(n_qubits, observable_name)
        param_ind=0
        gates = []
        wires = []
        params_cirquit = []
        parhere = []
        self.symbols = []
        self.target_reward = target_reward
        for gate_ind in trajectory:
            g = self.alphabet[str(int(gate_ind))]["gate"]
            wires.append(self.alphabet[str(int(gate_ind))]["wires"])
            if g in self.parametrized:  # assuming is one qubit unitary
                symbol = "x_{}".format(param_ind)
                self.symbols.append(symbol)
                params_cirquit.append(sympy.Symbol(self.symbols[-1]))
                param_ind += 1
                gates.append(g(params_cirquit[-1]))
                parhere.append(True)
            else:
                gates.append(g)
                parhere.append(False)
        self._wires = wires
        self._gates = gates
        self.parhere = parhere
        self.circuit = self.get_state(self.qubits)

    def get_state(self, qubits, params=None):
        circuit = cirq.Circuit()
        cc = []
        for q in qubits:
            cc.append(cirq.I.on(q))
        for ind, g in enumerate(self._gates):
            if len(self._wires[ind]) == 1:
                indqub = self._wires[ind][0]
                cc.append(g(qubits[indqub]))
            else:
                control, target = self._wires[ind]
                cc.append(g(qubits[control], qubits[target]))
        circuit.append(cc)
        if params is None:
            return circuit
        resolver = {k: v for k, v in zip(self.symbols, params)}
        return cirq.resolve_parameters(circuit, resolver)






if __name__ == "__main__":
    solver = CirqSolver()
    print(solver.run_circuit(np.array([0, 1, 2, 3, 4, 5, 4, 6, 5, 6, 7])))
