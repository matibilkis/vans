import numpy as np
import sympy
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
#
# """"
# This is a solver that we call Smart since it reduces
# the circuit according to some hand-crafted rules.
#
# For any other purpose it works like previous solvers:
# takes a list of actions (each action between 0 and len(solver.alphabet))
# and outputs energy and probabilities.
#
# Notice we do the circuit reduction each time we compute energy and probs.
# """"

#### we'll have to see to mute some CNOTS.

class CirqSmartSolver:
    def __init__(self, n_qubits=3, observable_name=None, target_reward=None):

        """
        observable_name:: specifies the hamiltonian; can be either string (if in templates, see load_observable function, a list
        or numpy array.

        target_reward:: minus the ground energy (or estimation), used as label for variational optimization.
        """

        self.name = "CirqSolver"
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.observable_name = observable_name


        ### create one_hot alphabet ####
        self.alphabet_gates = [cirq.CNOT, cirq.ry,cirq.rx(-np.pi/2), cirq.I]
        self.alphabet = []
        for ind, k in enumerate(range(5*self.n_qubits)): #5 accounts for 2 CNOTS and 3 other ops
            one_hot_gate = [-1]*5*self.n_qubits
            one_hot_gate[ind] = 1
            self.alphabet.append(one_hot_gate)

        ##### value to use as label for continuous optimization; this appears in variational model ####
        if target_reward is None:
            self.target_reward = self.n_qubits #mostly for the ising high transv fields.
        else:
            self.target_reward = target_reward

        self.observable, self.observable_matrix = self.load_observable(observable_name)

        ###indexed cnots total number n!/(n-2)! = n*(n-1) (if all connections are allowed)
        self.indexed_cnots = {}
        count=0
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control != target:
                    self.indexed_cnots[str(count)] = [control, target]
                    count+=1
        self.number_of_cnots = len(self.indexed_cnots)
        #int(np.math.factorial(self.n_qubits)/np.math.factorial(self.n_qubits -2))

    def load_observable(self, obs):
        """

        obs can either be a string, a list with cirq's gates or a matrix (array) """
        if obs is "W-state":  # then take projector on W state
            sq = 1 / np.sqrt(3)
            w_state = np.array([0, sq, sq, 0, sq, 0, 0, 0])
            w_proj = cirq.density_matrix_from_state_vector(w_state)
            observable = self.cirq_friendly_observable(w_proj)
            observable_matrix = w_proj

        elif obs == "Ising_High_TFields":
            observable = [cirq.X.on(q) for q in self.qubits] # -J \sum_{i} Z_i Z_{i+1} - g \sum_i X_i    when g>>J
            observable_matrix = cirq.unitary(cirq.Circuit(observable))

        elif obs is None:
            print("Define an observable please! Using identity meanwhile")
            observable = [cirq.I.on(q) for q in self.qubits] # -J \sum_{i} Z_i Z_{i+1} - g \sum_i X_i    when g>>J
            observable_matrix = cirq.unitary(cirq.Circuit(observable))
        else:
            print("loading observable, check estimation of ground state energy")
            if isinstance(obs, list):
                observable_matrix = cirq.unitary(cirq.Circuit(obs))
            else:
                observable = self.cirq_friendly_observable(obs)
                observable_matrix = w_proj
        return observable, observable_matrix


    def append_to_circuit(self, one_hot_gate, circuit, params):
        """
        appends to circuit the one_hot_gate;
        and if one_hot_gate it implies a rotation,
        appends to params a symbol"""

        for ind,inst in enumerate(one_hot_gate):
            if inst == 1: #this is faster than numpy.where
                if ind < self.number_of_cnots:
                    control, target = self.indexed_cnots[str(ind)]
                    circuit.append(self.alphabet_gates[0].on(self.qubits[control], self.qubits[target]))
                    return circuit, params
                elif self.number_of_cnots <= ind < 3*self.n_qubits:
                    new_param = "th_"+str(len(params))
                    params.append(new_param)
                    circuit.append(self.alphabet_gates[1](sympy.Symbol(new_param)).on(self.qubits[int(ind%self.n_qubits)]))
                    return circuit, params
                elif 3*self.n_qubits <= ind < 5*self.n_qubits:
                    circuit.append(self.alphabet_gates[int(np.trunc(ind/self.n_qubits))-1].on(self.qubits[int(ind%self.n_qubits)]))
                    return circuit, params
                else:
                    print("doing nothing! even not identity! careful")
                    return circuit, params



    def vansatz_keras_model(self, vansatz, observable):
        # notice observable may in general be expressed as linear combination
        # of different elements  on orthonormal basis obtained from tensor product
        # of Paulis. tf.math.reduce_sum is in charge of taking this linear combination.

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
        """

        takes as input vector with actions described as integer (given by RL agent)

        """
        wst = VAnsatzSmart(self.n_qubits,
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
        model.fit(x=w_input, y=w_output, batch_size=1, epochs=20, verbose=0)
        energy = float(np.squeeze(model.predict(w_input)))

        simulator = cirq.Simulator()
        result = simulator.simulate(wst.get_state(self.qubits,params=model.get_weights()[0]), qubit_order=self.qubits)
        probs = np.abs(result.final_state)**2
        return energy, probs

    def cirq_friendly_observable(self, obs):
        """this function takes a numpy array and converts it into pauli-based circuit"""
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
                    ot *= PAULI_BASIS_CIRQ[single_gate](qubits[qpos])*PAULI_BASIS_CIRQ[single_gate](qubits[qpos])
                else:
                    ot *= PAULI_BASIS_CIRQ[single_gate](qubits[qpos])
            if s < 3:
                unt.append(ot)
        return unt




class VAnsatzSmart(CirqSmartSolver):
    def __init__(self, n_qubits, observable_name, target_reward, trajectory):
        """

        takes as input a list with actions, each action being an integer between 0 and len(self.alphabet).

        .
        """
        super(VAnsatzSmart, self).__init__(n_qubits, observable_name)

        ws = [self.alphabet[int(g)] for g in trajectory]

        ws = self.check_and_recheck(ws)
        ws = self.detect_u3_and_reduce(ws)

        circuit = []
        params = []
        for g in ws:
            circuit, params = self.append_to_circuit(g,circuit, params)

        self.circuit = cirq.Circuit(circuit)

        #### check if there are qubits not called in the circuit  and add identity.
        #### (otherwise a problem with observable appears)

        effective_qubits = list(self.circuit.all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                self.circuit.append(cirq.I.on(k))
        self.symbols = params

    def get_state(self, qubits, params=None):
        if params is None:
            return self.circuit
        resolver = {k: v for k, v in zip(self.symbols, params)}
        return cirq.resolve_parameters(self.circuit, resolver)


    def check_two(self,w1,w2, c):

        """
        input::   w_1, w_2 one_hot_gate
                  c is an internal count used to interrupt the check_and_recheck if no more changes are done.
        output::  w_1, w_2 one_hot_gate in a "corrected form"
        """

        ind1 = np.where(np.array(w1) == 1)[0][0]
        ind2 = np.where(np.array(w2) == 1)[0][0]

        ## (i) both CNOTS, a) same CNOT, b) same targets, keep control on less index qubit first.
        if (ind1 < self.number_of_cnots) and (ind2 < self.number_of_cnots):
            if ind1 == ind2:
                w1[ind1] = -1
                w2[ind2] = -1
                control, target = self.indexed_cnots[str(ind1)]
                w1[self.number_of_cnots+2*self.n_qubits + control] = 1 #this is arbitrary
                w2[self.number_of_cnots+2*self.n_qubits + target] = 1 #this is arbitrary
                return w1, w2, c+1
            else:
                control1, target1 = self.indexed_cnots[str(ind1)]
                control2, target2 = self.indexed_cnots[str(ind2)]
                if target1 == target2:
                    if control1 > control2:
                        return w2, w1, c+1
                    else:
                        return w1,w2, c+1

        ## (ii) #both rz, replace second rotation by I
        if (self.number_of_cnots <= ind1 < self.number_of_cnots + self.n_qubits) and (self.number_of_cnots  <= ind2 < self.number_of_cnots + self.n_qubits):
            if ind1 == ind2:
                w2[ind1] = -1
                w2[self.number_of_cnots + 2*self.n_qubits + (ind2-self.number_of_cnots)%self.n_qubits] = 1
                return w1, w2, c+1
            elif ind1 > ind2: #put all gates on onequbit first
                return w2, w1, c+1

        ## (iii) P after CNOT (convention: put P before)
        if (ind1 < self.number_of_cnots) and (self.number_of_cnots + self.n_qubits <= ind2 < self.number_of_cnots + 2*self.n_qubits):
            control, target = self.indexed_cnots[str(ind1)]
            if target == (ind2- self.number_of_cnots)%self.n_qubits:
                return w2, w1, c+1

        ## (iv) rz after CNOT (convention: put rz before)
        if (ind1 < self.number_of_cnots) and (self.number_of_cnots <= ind2 < self.number_of_cnots + self.n_qubits):
            control, target = self.indexed_cnots[str(ind1)]
            if control == (ind2-self.number_of_cnots)%self.n_qubits:
                return w2, w1, c+1

        ##(v) move CNOTS as much to the right as possible
        if (ind1 < self.number_of_cnots) and (self.number_of_cnots <= ind2 < len(self.alphabet)): #
            if (ind2-self.number_of_cnots)%self.n_qubits not in self.indexed_cnots[str(ind1)]: #check if ind2 has something to do with control and targets of ind1..
                return w2, w1, c+1
            else:
                return w1, w2, c+1

        ## (vi) try to impose an order on single qubit gates
        if (self.number_of_cnots <= ind1 < len(self.alphabet)) and (self.number_of_cnots <= ind2 < len(self.alphabet)): #
            if (ind1-self.number_of_cnots)%self.n_qubits > (ind2-self.number_of_cnots)%self.n_qubits:
                return w2, w1, c+1
            else:
                return w1, w2, c+1
        else:
            return w1,w2, c


    def check_and_recheck(self, instructions, its=100, printing=False):
        """
        instructions:: collection of vectors of length |action_space|, eachone with information of only
        one qubit gate (or a cnot).

        its:: # of iterations (if no changes in one iteration, the loop is interrupted)
        prining:: boolean, visualize evolution - useful for debugging

        """
        ws_previous = instructions
        for k in range(its):
            count_changes = 0

            if printing:
                circuit = []
                params = []
                for g in ws_previous:
                    circuit, params = append_to_circuit(g,circuit, params)
                print(cirq.Circuit(circuit))
                print("************************************************************************************************************\n\n")

            ws_odd=[]
            for d in range(0,len(ws_previous),2):
                if d<len(ws_previous)-1:
                    w1 = ws_previous[d]
                    w2 = ws_previous[d+1]
                    wc1, wc2, count_changes = self.check_two(w1, w2, count_changes)
                    ws_odd.append(wc1)
                    ws_odd.append(wc2)
                else:
                    w1 = ws_previous[d]
                    ws_odd.append(w1)
            if printing:

                circuit = []
                params = []
                for g in ws_odd:
                    circuit, params = append_to_circuit(g,circuit, params)
                print(cirq.Circuit(circuit))
                print("************************************************************************************************************\n\n")


            ws_final = [ws_odd[0]]
            for d in range(1, len(ws_odd), 2):
                if d<len(ws_odd)-1:
                    w1 = ws_odd[d]
                    w2 = ws_odd[d+1]
                    wc1, wc2, count_changes = self.check_two(w1, w2, count_changes)
                    ws_final.append(wc1)
                    ws_final.append(wc2)
                else:
                    w1 = ws_odd[d]
                    ws_final.append(w1)
            ws_previous = ws_final
            ws_final = []
            if count_changes == 0:
                break
        return ws_previous


    def detect_u3_and_reduce(self,ws):
        """

            this function scans the circuit (represented as collection of one_hot_gates)
            and check if a pulse rz P rz P rz is found acting on a given qubit. If so
            kills previous consecutive 1-qubit gates.

        """
        cropped_circuit = np.array(ws)
        iss, jss = np.where(np.array(ws) == 1) #all indices with information of gate
        c=0
        internal_count_wire=0
        # u3_seq = np.array([2,3,2,3,2])
        u3_seq = np.array([0,1,0,1,0])#,2,1,2,1])

        for i,j in zip(iss, jss):
            if j>=self.number_of_cnots: #I don't want CNOTs
                while internal_count_wire == 0:
                    internal_count_wire +=1
                    qindfav = (j-self.number_of_cnots)%self.n_qubits #after the CNOTS, we have cycles of self.n_qubits one-qubit gates. qindfav then tells which is the qubit we are watching.
                    string_to_eval=[]
                    indexes_saving = []

                if ((j-self.number_of_cnots)%self.n_qubits == qindfav):
                    string_to_eval.append(int(np.trunc((j-self.number_of_cnots)/self.n_qubits)))
                    indexes_saving.append(i)
                    internal_count_wire+=1

                    if (i == iss[-1]): #it can happen that it's at one extreme
                        if internal_count_wire > 5:
                            if u3_seq in rolling(np.array(string_to_eval), 5): #this is u3
                                ind=0
                                for gind in indexes_saving: #erase everyone
                                    cropped_circuit[gind] = -1 #erase everyone
                                    if ind <5:
                                        cropped_circuit[gind][self.number_of_cnots +(u3_seq[ind]*self.n_qubits) + qindfav] = 1 #add each element of u3_seq in the corresponding position
                                        ind+=1
                                    else:
                                        cropped_circuit[gind][self.number_of_cnots + 2*self.n_qubits + qindfav] = 1 #add identity

                        internal_count_wire=0
                        string_to_eval=[]
                        indexes_saving = []
                else:
                    if internal_count_wire > 5:
                        if u3_seq in rolling(np.array(string_to_eval), 5): #this is u3
                            ind=0
                            for gind in indexes_saving: #erase everyone
                                cropped_circuit[gind] = -1 #erase everyone
                                if ind <5:
                                    cropped_circuit[gind][u3_seq[ind]*self.n_qubits + qindfav] = 1 #add each element of u3_seq
                                    ind+=1
                                else:
                                    cropped_circuit[gind][self.number_of_cnots + 2*self.n_qubits + qindfav] = 1 #add identity

                    internal_count_wire=0
                    string_to_eval=[]
                    indexes_saving = []
                c+=1
            else:
                internal_count_wire=0
                c=0
        return cropped_circuit

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



if __name__ == "__main__":
    solver = CirqSmartSolver(n_qubits=3,observable_name="Ising_High_TFields")
    # solver.run_circuit([0])

    solver.run_circuit(list(np.random.choice(range(15),30)))
    # print(solver.run_circuit(np.array([0, 1, 2, 3, 4, 5, 4, 6, 5, 6, 7])))
