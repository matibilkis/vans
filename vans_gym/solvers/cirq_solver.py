import numpy as np
import sympy
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq



class CirqSolver:
    def __init__(self, n_qubits=3, observable=None):
        self.name = "CirqSolver"
        self.n_qubits = n_qubits
        self.observable=observable #careful here!
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.alphabet = {"0":{"gate": cirq.X, "wires": [2]},
                            "1":{"gate": cirq.rz, "wires": [0]},
                            "2":{"gate": cirq.ry, "wires": [1]},
                            "3":{"gate": cirq.CNOT, "wires": [1,2]},#, "params":[np.pi]},
                            "4":{"gate": cirq.CNOT, "wires": [1,0]},#, "params":[np.pi]},
                            "5":{"gate": cirq.ry, "wires": [0]},
                            "6":{"gate":cirq.rz, "wires":[0]},#optimal sequence will be larger..
                            "7":{"gate": cirq.CNOT, "wires": [0,1]},#, "params":[np.pi]},
                           }

        self.parametrized = [cirq.rz, cirq.ry, cirq.rx]

        if observable is None:  # then take projector on W state
            sq = 1 / np.sqrt(3)
            w_state = np.array([0, sq, sq, 0, sq, 0, 0, 0])
            w_proj = cirq.density_matrix_from_state_vector(w_state)
            self.observable = self.cirq_friendly_observable(w_proj)



    def cirq_friendly_observable(self, obs):
        PAULI_BASIS = {
            'I': np.eye(2),
            'X': np.array([[0., 1.], [1., 0.]]),
            'Y': np.array([[0., -1j], [1j, 0.]]),
            'Z': np.diag([1., -1]),
        }

        pauli3 = cirq.linalg.operator_spaces.kron_bases(PAULI_BASIS, repeat=3)
        decomp = cirq.linalg.operator_spaces.expand_matrix_in_orthogonal_basis(obs, pauli3) #notice it's not required to be orthonormal!

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
            s=0
            ot=float(coeff)
            for qpos, single_gate in enumerate(gate_name):
                if single_gate == "I":
                    ot*=PAULI_BASIS_CIRQ[single_gate](self.qubits[qpos])*PAULI_BASIS_CIRQ[single_gate](self.qubits[qpos])
                else:
                    ot*=PAULI_BASIS_CIRQ[single_gate](self.qubits[qpos])
            if s<3:
                unt.append(ot)
        return unt
    #
    #

    def vansatz_keras_model(self, vansatz, observable):
        #notice observable may in general be expressed as linear combination
        #of different elements  on orthonormal basis obtained from tensor product
        #of SU(2) generators. tf.math.reduce_sum is in charge of taking this linear combination.
        circuit_input = tf.keras.Input(shape=(), dtype=tf.string)
        output = tfq.layers.Expectation()(
                circuit_input,
                symbol_names=vansatz.symbols,
                operators=tfq.convert_to_tensor([observable]),
                initializer=tf.keras.initializers.RandomNormal())

        output = tf.math.reduce_sum(output, axis=-1, keepdims=True)

        model = tf.keras.Model(inputs=circuit_input, outputs=output)
        adam = tf.keras.optimizers.Adam(learning_rate=0.1)
        model.compile(optimizer=adam, loss='mse')
        return model

    def run_circuit(self, list_ops):
        trajectory = list_ops#
        wst = VAnsatz(trajectory)
        model = self.vansatz_keras_model(wst, self.observable)
        w_input = tfq.convert_to_tensor([wst.circuit])
        w_output = tf.ones((1,1)) #in case of W_state we want fidelity 1.
        model.fit(x=w_input, y=w_output, batch_size=1, epochs=50,
                    verbose=0)
        energy = float(np.squeeze(model.predict(w_input)))

        simulator = cirq.Simulator()
        result = simulator.simulate(wst.get_state(self.qubits,params=model.get_weights()[0]), qubit_order=self.qubits)
        probs = np.abs(result.final_state)**2
        return energy, probs



class VAnsatz(CirqSolver):
    def __init__(self, trajectory):
        super(VAnsatz, self).__init__()
        param_ind=0
        gates=[]
        wires=[]
        params_cirquit=[]
        parhere=[]
        self.symbols=[]
        for gate_ind in trajectory:
            g = self.alphabet[str(int(gate_ind))]["gate"]
            wires.append(self.alphabet[str(int(gate_ind))]["wires"])
            if g in self.parametrized: #assuming is one qubit unitary
                symbol = "x_{}".format(param_ind)
                self.symbols.append(symbol)
                params_cirquit.append(sympy.Symbol(self.symbols[-1]))
                param_ind+=1
                gates.append(g(params_cirquit[-1]))
                parhere.append(True)
            else:
                gates.append(g)
                parhere.append(False)
        self.wires = wires
        self._gates=gates
        self.parhere =parhere
        self.circuit=self.get_state(self.qubits)

    def get_state(self, qubits, params=None):
        circuit = cirq.Circuit()
        cc=[]
        for ind, g in enumerate(self._gates):
            if len(self.wires[ind])==1:
                indqub = self.wires[ind][0]
                cc.append(g(qubits[indqub]))
            else:
                control, target = self.wires[ind]
                cc.append(g(qubits[control], qubits[target]))
        circuit.append(cc)
        if params is None:
            return circuit
        resolver = {k: v for k, v in zip(self.symbols, params)}
        return cirq.resolve_parameters(circuit, resolver)




if __name__ == "__main__":
    solver = CirqSolver()
    print(solver.run_circuit(np.array([0,1,2,3,4,5,4,6,5,6,7])))
