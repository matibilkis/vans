import numpy as np
import sympy
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

class CirqSolverR:
    def __init__(self, n_qubits=3, observable_name=None, ground_state_energy=None, qlr=0.01, qepochs=100,display_progress=0):

        """
        observable_name:: specifies the hamiltonian; can be either string (if in templates, see load_observable function, a list
        or numpy array.

        target_reward:: minus the ground energy (or estimation), used as label for variational optimization.
        display_progress:: 0 or 1 (no, yes)
        """

        self.name = "CirqSolver"
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.observable_name = observable_name

        # Value to use as label for continuous optimization; this appears in variational model
        if ground_state_energy is None:
            self.ground_state_energy = self.n_qubits  # mostly for the ising high transv fields.
        else:
            self.ground_state_energy = ground_state_energy

        self.observable = self.load_observable(observable_name) #Ising hamiltonian with list of pauli_gates
        self.qlr = qlr
        self.qepochs=qepochs
        self.display_progress=display_progress
        # Indexed cnots total number n!/(n-2)! = n*(n-1) (if all connections are allowed)
        self.indexed_cnots = {}
        count = 0
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control != target:
                    self.indexed_cnots[str(count)] = [control, target]
                    count += 1
        self.number_of_cnots = len(self.indexed_cnots)
        #int(np.math.factorial(self.n_qubits)/np.math.factorial(self.n_qubits -2))

        # Create one_hot a+lphabet
        self.alphabet_gates = [cirq.CNOT, cirq.rz, cirq.rx(-np.pi/2), cirq.I]
        self.alphabet = []

        alphabet_length = self.number_of_cnots + (len(self.alphabet_gates)-1)*self.n_qubits
        for ind, k in enumerate(range(self.number_of_cnots + (len(self.alphabet_gates)-1)*self.n_qubits)): #one hot encoding
            one_hot_gate = [-1.]*alphabet_length
            one_hot_gate[ind] = 1.
            self.alphabet.append(one_hot_gate)

        self.final_params = []
    def index_meaning(self,index):
        if index<self.number_of_cnots:
            print("cnot: ",self.indexed_cnots[str(index)])
            return
        else:
            print(self.alphabet_gates[1:][int((index-self.number_of_cnots)/self.n_qubits)], "on qubit: ",(index-self.number_of_cnots)%self.n_qubits)
            return

    def load_observable(self, obs,g=1, J=0):
        """
        obs can either be a string, a list with cirq's gates or a matrix (array)
        """
        if obs == "Ising_":
            observable = [g*cirq.X.on(q) for q in self.qubits] # -J \sum_{i} Z_i Z_{i+1} - g \sum_i X_i    when g>>J
            for q in range(len(self.qubits)):
                observable.append(J*cirq.Z.on(self.qubits[q])*cirq.Z.on(self.qubits[(q+1)%len(self.qubits)]))
        elif obs == "EasyIsing_":
            observable = [g*cirq.Z.on(q) for q in self.qubits] # -J \sum_{i} Z_i Z_{i+1} - g \sum_i X_i    when g>>J
            for q in range(len(self.qubits)):
                observable.append(J*cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
        else:
            print("check previous versions to load other observables.")
        return observable

    def append_to_circuit(self, one_hot_gate, circuit, params):
        """
        appends to circuit the one_hot_gate;
        and if one_hot_gate it implies a rotation,
        appends to params a symbol"""

        for ind,inst in enumerate(one_hot_gate):
            if inst == 1:  # this is faster than numpy.where
                if ind < self.number_of_cnots:
                    control, target = self.indexed_cnots[str(ind)]
                    circuit.append(self.alphabet_gates[0].on(self.qubits[control], self.qubits[target]))
                    return circuit, params
                elif self.number_of_cnots <= ind < self.number_of_cnots + self.n_qubits:
                    new_param = "th_"+str(len(params))
                    params.append(new_param)
                    circuit.append(self.alphabet_gates[1](sympy.Symbol(new_param)).on(self.qubits[int(ind%self.n_qubits)]))
                    return circuit, params
                elif self.number_of_cnots + self.n_qubits <= ind < self.number_of_cnots + 2*self.n_qubits:
                    circuit.append(self.alphabet_gates[2].on(self.qubits[int(ind%self.n_qubits)]))
                    return circuit, params
                elif self.number_of_cnots + 2*self.n_qubits <= ind < self.number_of_cnots+3*self.number_of_cnots:
                    circuit.append(self.alphabet_gates[3].on(self.qubits[int(ind%self.n_qubits)]))
                    return circuit, params
                else:
                    print("doing nothing! even not identity! careful")
                    return circuit, params


    def TFQ_model(self, symbols):
        circuit_input = tf.keras.Input(shape=(), dtype=tf.string)
        output = tfq.layers.Expectation()(
                circuit_input,
                symbol_names=symbols,
                operators=tfq.convert_to_tensor([self.observable]),
                initializer=tf.keras.initializers.RandomNormal()) #this is not strictly necessary.
        model = tf.keras.Model(inputs=circuit_input, outputs=output)
        adam = tf.keras.optimizers.Adam(learning_rate=self.qlr)
        model.compile(optimizer=adam, loss='mse')
        return model

    def give_circuit(self, lista,one_hot=False):
        if not one_hot:
            circuit, symbols = [], []
            for k in lista:
                circuit, symbols = self.append_to_circuit(self.alphabet[k],circuit,symbols)
        else:
            circuit, symbols = [], []
            for k in lista:
                circuit, symbols = self.append_to_circuit(k,circuit,symbols)
        circuit = cirq.Circuit(circuit)
        return circuit, symbols

    def run_circuit(self, gates_index, sim_q_state=False):
        """
        takes as input vector with actions described as integer (given by RL agent),
        and outputsthe energy of that circuit (w.r.t self.observable)
            """
        circuit, symbols = [], []
        for k in gates_index:
            circuit, symbols = self.append_to_circuit(self.alphabet[int(k)],circuit,symbols)

        circuit = cirq.Circuit(circuit)
        effective_qubits = list(circuit.all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                circuit.append(cirq.I.on(k))

        tfqcircuit = tfq.convert_to_tensor([circuit])
        if len(symbols) == 0:
            expval = tfq.layers.Expectation()(
                                            tfqcircuit,
                                            operators=tfq.convert_to_tensor([self.observable]))
            energy = np.float32(np.squeeze(tf.math.reduce_sum(expval, axis=-1, keepdims=True)))
            self.final_params = []

        else:
            model = self.TFQ_model(symbols)
            qoutput = tf.ones((1, 1))*self.ground_state_energy
            model.fit(x=tfqcircuit, y=qoutput, batch_size=1, epochs=self.qepochs, verbose=self.display_progress)
            energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))
            self.final_params = [np.squeeze(k.numpy()) for k in model.trainable_variables]

        #if sim_q_state:
            #simulator = cirq.Simulator()
            #result = simulator.simulate(circuit, qubit_order=self.qubits)
            #probs = np.abs(result.final_state)**2
            #return energy, probs
        return energy

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class Checker():
    def __init__(self, solver):

        """
        takes a trajectory drawn from solver alphabet (solver is an object), trajectory is list of integer indeces
        from 0 to len(solver.alphabet)-1
        """

        self.solver = solver


    def correct_trajectory(self, trajectory): #trajectory = sequence

        OneHotGates = self.convert_to_One_Hot(trajectory)
        traj_without_id = self.delete_identities(OneHotGates)
        OneHotGates = self.convert_to_One_Hot(traj_without_id)
        ###this may be crrected.. or looped many times..###
        oh = self.check_and_recheck(OneHotGates)
        for k in range(2):
            oh = self.detect_u3_and_reduce(oh)
            oh = self.convert_to_One_Hot(self.delete_identities(oh))
            if len(oh)==0:
                break
            oh = self.check_and_recheck(oh)
        final_traj=self.delete_identities(oh)
        return final_traj

    def convert_to_One_Hot(self, trajectory):
        OneHotGates=[]
        for g in trajectory:
            if g == -1.:
                break
            else:
                OneHotGates.append(self.solver.alphabet[int(g)])
                #later on i'll loop on this!
        return OneHotGates

    def check_two(self,ww1,ww2, c=0):
        """
        input::   w_1, w_2 one_hot_gates
                  c is an internal count used to interrupt the check_and_recheck if no more changes are done (used in an outer loop)

        output::  w_1, w_2 one_hot_gate in a "corrected form"
        """
        ind1 = np.where(np.array(ww1) == 1.)[0][0]
        ind2 = np.where(np.array(ww2) == 1.)[0][0]

        w1 = ww1.copy()
        w2 = ww2.copy()

        ## (i) both CNOTS, a) same CNOT, b) same targets, keep control on less index qubit first.
        if (ind1 < self.solver.number_of_cnots) and (ind2 < self.solver.number_of_cnots):
            if ind1 == ind2:
                w1[ind1] = -1.
                w2[ind2] = -1.
                control, target = self.solver.indexed_cnots[str(ind1)]
                w1[self.solver.number_of_cnots+(2*self.solver.n_qubits) + control] = 1. #change for identity in control
                w2[self.solver.number_of_cnots+(2*self.solver.n_qubits) + target] = 1. #change for identity in target
                #print("i1")
                return w1, w2, c+1
            else:
                control1, target1 = self.solver.indexed_cnots[str(ind1)]
                control2, target2 = self.solver.indexed_cnots[str(ind2)]
                if target1 == target2:
                    if control1 > control2:
                        #print("i2")
                        return w2, w1, c+1
                    else:
                        return w1, w2, c

                else:
                    return w1, w2, c

         ## (ii) #both rz, replace second rotation by I
        if (self.solver.number_of_cnots <= ind1 < self.solver.number_of_cnots + self.solver.n_qubits) and (self.solver.number_of_cnots  <= ind2 < self.solver.number_of_cnots + self.solver.n_qubits):
            if ind1 == ind2:
                w2[ind1] = -1.
                w2[self.solver.number_of_cnots + 2*self.solver.n_qubits + (ind2-self.solver.number_of_cnots)%self.solver.n_qubits] = 1.
                #print("i3")
                return w1, w2, c+1
            elif ind1 > ind2: #put all gates on onequbit first
                #print("i4")
                return w2, w1, c+1
            else:
                return w1, w2, c

#         ## (iii) P after CNOT (convention: put P before)
        if (ind1 < self.solver.number_of_cnots) and ((self.solver.number_of_cnots + self.solver.n_qubits) <= ind2 < self.solver.number_of_cnots + 2*self.solver.n_qubits):
            control, target = self.solver.indexed_cnots[str(ind1)]
            if target == (ind2- self.solver.number_of_cnots)%self.solver.n_qubits:
                #print("i5")
                ##print(ind1, ind2)
                ##print("target, ",target)
                ##print("w1,w2: ",w1, w2)
                return w2, w1, c+1
            else:
                return w1, w2, c

        ## (iv) rz after CNOT (convention: put rz before)
        if (ind1 < self.solver.number_of_cnots) and (self.solver.number_of_cnots <= ind2 < self.solver.number_of_cnots + self.solver.n_qubits):
            control, target = self.solver.indexed_cnots[str(ind1)]
            if control == (ind2-self.solver.number_of_cnots)%self.solver.n_qubits:
                #print("i6")
                ##print(control)
                ##print("indexes", ind1,ind2)
                ##print(w1, w2)
                return w2, w1, c+1
            else:
                return w1, w2, c

        ##(v) move CNOTS as much to the right as possible (not considering identity)
        if (ind1 < self.solver.number_of_cnots) and (self.solver.number_of_cnots <= ind2 < self.solver.number_of_cnots + 2*self.solver.n_qubits): #
            if (ind2-self.solver.number_of_cnots)%self.solver.n_qubits not in self.solver.indexed_cnots[str(ind1)]: #check if ind2 has something to do with control and targets of ind1..
                #print("i7 moving cnots")
                return w2, w1, c+1
            else:
                return w1, w2, c

        ## (vi) to impose an order on single qubit gates
        if (self.solver.number_of_cnots <= ind1 < len(self.solver.alphabet)) and (self.solver.number_of_cnots <= ind2 < len(self.solver.alphabet)): #
            if (ind1-self.solver.number_of_cnots)%self.solver.n_qubits > (ind2-self.solver.number_of_cnots)%self.solver.n_qubits:
                return w2, w1, c+1
                #print("i8")
            elif ((ind1-self.solver.number_of_cnots)%self.solver.n_qubits == (ind2-self.solver.number_of_cnots)%self.solver.n_qubits) and (self.solver.number_of_cnots <= ind2 < self.solver.number_of_cnots + 2*self.solver.n_qubits) and (self.solver.number_of_cnots + 2*self.solver.n_qubits <= ind1):
                return w2,w1,c+1 ###just move identiy to the right if possible, for a single qubit
            else:
                return w1, w2, c

        else:
            return w1, w2, c

    def detect_u3_and_reduce(self,OneHotGates):
        """
            this function scans the circuit (represented as collection of one_hot_gates)
            and check if a pulse rz P rz P rz is found acting on a given qubit. If so
            kills previous consecutive 1-qubit gates.
        """

        cropped_circuit = np.array(OneHotGates)
        iss, jss = np.where(np.array(OneHotGates) == 1.)  # all indices with information of gate
        c=0
        internal_count_wire=0
        u3_seq = np.array([0, 1, 0, 1, 0]) #rz, p, rz, p, rz

        for i,j in zip(iss, jss):
            if j>=self.solver.number_of_cnots: #I don't want CNOTs
                while internal_count_wire == 0:
                    internal_count_wire +=1
                    qindfav = (j-self.solver.number_of_cnots)%self.solver.n_qubits  # after the CNOTS, we have cycles of self.n_qubits one-qubit gates. qindfav then tells which is the qubit we are watching.
                    string_to_eval=[]
                    indexes_saving = []

                if (j-self.solver.number_of_cnots) % self.solver.n_qubits == qindfav:
                    string_to_eval.append(int(np.trunc((j-self.solver.number_of_cnots)/self.solver.n_qubits)))
                    indexes_saving.append(i)
                    internal_count_wire+=1

                    if i == iss[-1]:  # it can happen that it's at one extreme
                        if internal_count_wire > 5:
                            if u3_seq in rolling(np.array(string_to_eval), 5):  # this is u3
                                ind=0
                                for gind in indexes_saving:
                                    cropped_circuit[gind] = -1  # erase everyone
                                    if ind < 5:
                                        cropped_circuit[gind][self.solver.number_of_cnots +(u3_seq[ind]*self.solver.n_qubits) + qindfav] = 1  # add each element of u3_seq in the corresponding position
                                        ind += 1
                                    else:
                                        cropped_circuit[gind][self.solver.number_of_cnots + 2*self.solver.n_qubits + qindfav] = 1  # add identity

                        internal_count_wire=0
                        string_to_eval=[]
                        indexes_saving = []
                else:
                    if internal_count_wire > 5:
                        if u3_seq in rolling(np.array(string_to_eval), 5):  # this is u3
                            ind = 0
                            for gind in indexes_saving:  # erase everyone
                                cropped_circuit[gind] = -1  # erase everyone
                                if ind < 5:
                                    cropped_circuit[gind][u3_seq[ind]*self.solver.n_qubits + qindfav] = 1  # add each element of u3_seq
                                    ind += 1
                                else:
                                    cropped_circuit[gind][self.solver.number_of_cnots + 2*self.solver.n_qubits + qindfav] = 1  # add identity

                    internal_count_wire=0
                    string_to_eval=[]
                    indexes_saving = []
                c += 1
            else:
                internal_count_wire=0
                c = 0
        return cropped_circuit


    def check_and_recheck(self, instructions, its=100):
        """
        instructions:: collection of One Hot gates.

        its:: # of iterations (if no changes in one iteration, the loop is interrupted)
        prining:: boolean, visualize evolution - useful for debugging

        """
        ws_previous = instructions
        for k in range(its):
            count_changes = 0

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

    def delete_identities(self, OneHotGates):
        gates_integers=[]
        for k in OneHotGates:
            one = np.where(np.array(k)==1.)[0][0]
            if one < self.solver.number_of_cnots + 2*self.solver.n_qubits:
                gates_integers.append(one)
        return np.array(gates_integers)#.astype(np.float32)
