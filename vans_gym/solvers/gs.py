import numpy as np
import sympy
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
#from IPython.display import SVG, display
#from cirq.contrib.svg import SVGCircuit


class GeneticSolver:
    def __init__(self, n_qubits=3, qlr=0.01, qepochs=100,verbose=0, g=1, J=0, noises={}):

        """"solver with n**2 possible actions: n(n-1) CNOTS + n 1-qubit unitary"""
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.lower_bound_Eg = -2*self.n_qubits

        self.qlr = qlr
        self.qepochs=qepochs
        self.verbose=verbose


        self.indexed_cnots = {}
        self.cnots_index = {}
        count = 0
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control != target:
                    self.indexed_cnots[str(count)] = [control, target]
                    self.cnots_index[str([control,target])] = count
                    count += 1
        self.number_of_cnots = len(self.indexed_cnots)

        self.final_params = []
        self.parametrized_unitary = [cirq.rz, cirq.rx, cirq.rz]
        self.noises=noises
        self.noise_level = self.noises["depolarizing"]

        self.observable=self.ising_obs(g=g, J=J)
        self.resolver = {}
        self.new_resolver = {} #this temporarly stores initialized parameters of identity resolution
        self.lowest_energy_found = -.1
        self.best_circuit_found = []
        self.best_resolver_found = {}

        self.expectation_layer = tfq.layers.Expectation(backend=cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.001)))


    def ising_obs(self, g=1, J=0):
        # -  \Gamma/2 \sum_i Z_i - J/2 \sum_{i} X_i X_{i+1}    (S_i = \Sigma_i/2; ej S_z = Z/2, S_x = X/2)
        ### analytic solution https://sci-hub.tw/https://www.sciencedirect.com/science/article/abs/pii/0003491670902708?via%3Dihub
        observable = [-float(0.5*g)*cirq.Z.on(q) for q in self.qubits]
        for q in range(len(self.qubits)):
            observable.append(-float(0.5*J)*cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
        #### E_0 = -\Gamma/2 \sum_k \Lambda_k , with \Lambda_k = \sqrt{ 1 + \lambda^{2}  + 2 \lambda \cos(k)};
        ### k = -N/2, ... , 0 ,... N/2-1 if N even
        #### k = -(N-1)/2, ... 0 , ... (N-1)/2 if N odd
        if self.n_qubits%2 == 0:
            val = -self.n_qubits/2
        else:
            val = -(self.n_qubits-1)/2
        values_q = []
        for k in range(2*self.n_qubits):
            values_q.append(val)
            val += 1/2
        ###soething wrong here.
        self.ground_energy = -(0.5*g)*np.sum(np.sqrt([1+(J/(2*g))**2 - (np.cos(2*np.pi*q/self.n_qubits)*(J/g)) for q in values_q]))
        return observable

    def index_meaning(self,index):
        if index<self.number_of_cnots:
            print("cnot: ",self.indexed_cnots[str(index)])
            return
        else:
            print("1-qubit unitary on: ",(index-self.number_of_cnots)%self.n_qubits)
            return

    def append_to_circuit(self, ind, circuit, params, new_index=False):
        """
        appends to circuit the index of the gate;
        and if one_hot_gate implies a rotation,
        appends to params a symbol
        """
        if ind < self.number_of_cnots:
            control, target = self.indexed_cnots[str(ind)]
            circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            return circuit, params
        else:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(3),self.parametrized_unitary):
                new_param = "th_"+str(len(params))
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
            return circuit, params

    def give_circuit(self, lista,one_hot=False):
        circuit, symbols = [], []
        for k in lista:
            circuit, symbols = self.append_to_circuit(k,circuit,symbols)
        circuit = cirq.Circuit(circuit)
        return circuit, symbols


    def resolution_2cnots(self, q1, q2):
        u1 = self.number_of_cnots + q1
        u2 = self.number_of_cnots + q2
        cnot = self.cnots_index[str([q1,q2])]
        return [cnot, u1, u2, cnot]

    def resolution_1qubit(self, q):
        u1 = self.number_of_cnots + q
        return [u1]


    def dressed_cnot(self,q1,q2):
        u1 = self.number_of_cnots + q1
        u2 = self.number_of_cnots + q2
        cnot = self.cnots_index[str([q1,q2])]
        u3 = self.number_of_cnots + q1
        u4 = self.number_of_cnots + q2
        return [u1,u2,cnot,u3,u4]

    def dressed_ansatz(self, layers=1):
        c=[]
        for layer in range(layers):
            qubits = list(range(self.n_qubits))
            qdeph = qubits[layers:]
            for q in qubits[:layers]:
                qdeph.append(q)
            for ind1, ind2 in zip(qubits,qdeph):
                for k in self.dressed_cnot(ind1,ind2):
                    c.append(k)
        return c


    def prepare_circuit_insertion(self,gates_index, block_to_insert, insertion_index):
        """gates_index is a vector with integer entries, each one describing a gate
            block_to_insert is block of unitaries to insert at index insertion
        """
        circuit = cirq.Circuit()
        idx_circuit=[]
        symbols = []
        new_symbols = []
        new_resolver = {}

        if gates_index == []:
            indices = [-1]
        else:
            indices = gates_index
        for ind, g in enumerate(indices):
            #### insert new block ####
            if ind == insertion_index:
                for gate in block_to_insert:
                    idx_circuit.append(gate)
                    if gate < self.number_of_cnots:
                        control, target = self.indexed_cnots[str(gate)]
                        circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
                    else:
                        qubit = self.qubits[(gate-self.number_of_cnots)%self.n_qubits]
                        for par, gateblack in zip(range(3),self.parametrized_unitary):
                            new_symbol = "New_th_"+str(len(new_symbols))
                            new_symbols.append(new_symbol)
                            new_resolver[new_symbol] = np.random.uniform(-.1,.1) #rotation around epsilon... we can do it better afterwards
                            circuit.append(gateblack(sympy.Symbol(new_symbol)).on(qubit))
            if 0<= g < self.number_of_cnots:
                idx_circuit.append(g)
                control, target = self.indexed_cnots[str(g)]
                circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            elif g>= self.number_of_cnots:

                idx_circuit.append(g)
                qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
                for par, gate in zip(range(3),self.parametrized_unitary):
                    new_symbol = "th_"+str(len(symbols))
                    symbols.append(new_symbol)
                    circuit.append(gate(sympy.Symbol(new_symbol)).on(qubit))
                    if not new_symbol in self.resolver.keys(): #this is in case it's the first time. Careful when deleting !
                        self.resolver[new_symbol] = np.random.uniform(-np.pi, np.pi)

        ### add identity for TFQ tocompute correctily expected value####
        effective_qubits = list(circuit.all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                circuit.append(cirq.I.on(k))
        self.new_resolver = new_resolver
        variables = [symbols, new_symbols]
        return circuit, variables#, idx_circuit



    def TFQ_model(self, symbols, lr=None):
        circuit_input = tf.keras.Input(shape=(), dtype=tf.string)
        output = self.expectation_layer(
                circuit_input,
                symbol_names=symbols,
                operators=tfq.convert_to_tensor([self.observable]),
                initializer=tf.keras.initializers.RandomNormal()) #we may change this!!!

        model = tf.keras.Model(inputs=circuit_input, outputs=output)
        if lr is None:
            adam = tf.keras.optimizers.Adam(learning_rate=self.qlr)
        else:
            adam = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=adam, loss='mse')
        return model

    def initialize_model_insertion(self, variables):
        ### initialize model with parameters from previous model (describer by variables[0]) --> values in self.resolver
        ###(for the already-optimized ones), and close to identity for the block added, described by variables[1], whose values are in self.new_resolver

        symbols, new_symbols = variables
        circuit_symbols = []
        init_params = []
        for j in symbols:
            circuit_symbols.append(j)
            init_params.append(self.resolver[str(j)])#+ np.random.uniform(-.01,.01)) if you want to perturbate previous parameters..
        for k in new_symbols:
            circuit_symbols.append(k)
            init_params.append(self.new_resolver[str(k)])

        model = self.TFQ_model(circuit_symbols)
        model.trainable_variables[0].assign(tf.convert_to_tensor(init_params)) #initialize parameters of model (continuous parameters of uniraries)
        #with the corresponding values
        return model

    def run_circuit_from_index(self, gates_index, hyperparameters=None):
        """
        takes as input vector with actions described as integer
        and outputsthe energy of that circuit (w.r.t self.observable)

        hyperparameters = [epoch, lr]
        """
        ### create a vector with the gates on the corresponding qubit(s)
        circuit, symbols = self.give_circuit(gates_index)

        ### this is because each qubit should be "activated" in TFQ to do the optimization (if the observable has support on this qubit as well and you don't add I then error)
        effective_qubits = list(circuit.all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                circuit.append(cirq.I.on(k))

        tfqcircuit = tfq.convert_to_tensor([circuit])
        if len(symbols) == 0:

            expval = self.expectation_layer(tfqcircuit,
                                            operators=tfq.convert_to_tensor([self.observable]))
            energy = np.float32(np.squeeze(tf.math.reduce_sum(expval, axis=-1, keepdims=True)))
            final_params = []
            resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
        else:
            if hyperparameters is None:
                model = self.TFQ_model(symbols)
                qoutput = tf.ones((1, 1))*self.lower_bound_Eg
                print("about to fit!")

                model.fit(x=tfqcircuit, y=qoutput, batch_size=1, epochs=self.qepochs,verbose=self.verbose,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)])
                # model.fit(x=tfqcircuit, y=qoutput, batch_size=1, steps_per_epoch=1,epochs=self.qepochs, verbose=self.verbose,workers=1)#, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)])#, MemoryCallback()])
                energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))
                final_params = model.trainable_variables[0].numpy()
                resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
            else:
                model = self.TFQ_model(symbols, hyperparameters[1])
                qoutput = tf.ones((1, 1))*self.lower_bound_Eg
                print("about to fit!")

                model.fit(x=tfqcircuit, y=qoutput, batch_size=1, epochs=self.qepochs,verbose=self.verbose,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)])
                # model.fit(x=tfqcircuit, y=qoutput, batch_size=1, steps_per_epoch=1, epochs=hyperparameters[0], verbose=self.verbose,workers=1)#,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)])#,MemoryCallback()])
                energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))
                final_params = model.trainable_variables[0].numpy()
                resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
        #self.current_circuit = gates_index
        self.resolver = resolver
        if self.accept_modification(energy):
            self.lowest_energy_found = energy
            self.best_circuit_found = gates_index
            self.best_resolver_found = resolver
        return gates_index, resolver, energy


    def accept_modification(self, energy):
        return np.abs(energy)/np.abs(self.lowest_energy_found) > .98


    def optimize_and_update(self, gates_index, model, circuit,variables,insertion_index_loaded, block_to_insert):

        effective_qubits = list(circuit.all_qubits())
        q=0
        for k in self.qubits:#che, lo que no estoy
            if k not in effective_qubits:
                circuit.append(cirq.I.on(k))
                q+=1
        if q == self.n_qubits:
            circuit.append(cirq.rz(sympy.Symbol("dummy")).on(self.qubits[0])) #hopefully you won't accept this, but in case you do, then it's better since it simplifies...


        tfqcircuit = tfq.convert_to_tensor([circuit])
        qoutput = tf.ones((1, 1))*self.lower_bound_Eg
        model.fit(x=tfqcircuit, y=qoutput, batch_size=1, epochs=self.qepochs,verbose=self.verbose,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)])

        energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))

        if self.accept_modification(energy):

            #### if we accept the new configuration, then we update the resolver merging both symbols and new_symbols into self.resolver
            symbols, new_symbols = variables

            for ind,k in enumerate(symbols):
                self.resolver[k] = model.trainable_variables[0].numpy()[ind]

            for indnew,knew in enumerate(new_symbols):
                self.new_resolver[knew] = model.trainable_variables[0].numpy()[len(symbols)+indnew]

            final_symbols = []
            old_solver = []
            old_added = []

            final_resolver = {}
            new_circuit = []
            for ind, g in enumerate( gates_index):
                #### insert new block ####
                if ind == insertion_index_loaded:
                    for gate in block_to_insert:
                        new_circuit.append(gate)
                        if gate < self.number_of_cnots:
                            pass
                        else:
                            for par, gateblock in zip(range(3),self.parametrized_unitary):

                                var1 = "New_th_"+str(len(old_added))
                                old_added.append(var1)

                                var2 = "th_"+str(len(final_symbols))
                                final_symbols.append(var2)
                                final_resolver[var2] = self.new_resolver[var1] #

                if g < self.number_of_cnots:
                    new_circuit.append(g)
                    pass
                else:
                    new_circuit.append(g)
                    for par, gate in zip(range(3),self.parametrized_unitary):
                        var3 = "th_"+str(len(old_solver))
                        old_solver.append(var3)

                        var4 = "th_"+str(len(final_symbols))
                        final_symbols.append(var4)
                        final_resolver[var4] = self.resolver[var3]

            self.resolver = final_resolver
            #self.current_circuit = new_circuit #### now the current circuit is the better one! otherwise you keep the previous (from self.run_circuit_from_index)
            self.best_circuit_found = new_circuit
            self.lowest_energy_found = energy
            self.best_resolver_found = final_resolver
            return new_circuit, self.resolver, energy, True
        else:
            return gates_index, self.resolver, self.lowest_energy_found, False

    def kill_one_unitary(self, gates_index, resolver, energy):
        """
        this function takes circuit as described by gates_index (sequence of integers)
        and returns when possible, a circuit, resolver, energy with one single-qubit unitary less.
        """

        circuit_proposals=[] #storing all good candidates.
        circuit_proposals_energies=[]
        for j in gates_index:
            indexed_prop=[]

            prop=cirq.Circuit()
            checking = False
            ko=0
            to_pop=[]

            for k in gates_index:
                if k < self.number_of_cnots:
                    indexed_prop.append(k)
                    control, target = self.indexed_cnots[str(k)]
                    prop.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
                else:
                    if k != j:
                        indexed_prop.append(k)
                        qubit = self.qubits[(k-self.number_of_cnots)%self.n_qubits]
                        for par, gate in zip(range(3),self.parametrized_unitary):
                            new_param = "th_"+str(ko)
                            ko+=1
                            prop.append(gate(sympy.Symbol(new_param)).on(qubit))
                    else:
                        checking=True
                        for i in range(3):
                            to_pop.append("th_"+str(ko))
                            ko+=1
            if checking is True:
                nr = resolver.copy()
                for p in to_pop:
                    nr.pop(p)

                effective_qubits = list(prop.all_qubits())
                for k in self.qubits:
                    if k not in effective_qubits:
                        prop.append(cirq.I.on(k))

                tfqcircuit = tfq.convert_to_tensor([cirq.resolve_parameters(prop, nr)]) ###resolver parameters !!!
                expval=self.expectation_layer(tfqcircuit,
                            operators=tfq.convert_to_tensor([self.observable]))
                new_energy = np.float32(np.squeeze(tf.math.reduce_sum(expval, axis=-1, keepdims=True)))

                if self.accept_modification(new_energy):
                    ordered_resolver = {}
                    for ind,k in enumerate(nr.values()):
                        ordered_resolver["th_"+str(ind)] = k
                    circuit_proposals.append([indexed_prop,ordered_resolver,new_energy])
                    circuit_proposals_energies.append(new_energy)
        if len(circuit_proposals)>0:
            favourite = np.random.choice(len(circuit_proposals))
            short_circuit, resolver, energy = circuit_proposals[favourite]
            #self.current_circuit = short_circuit
            self.resolver = resolver
            self.best_resolver_found = resolver
            self.best_circuit_found = short_circuit
            self.lowest_energy_found = circuit_proposals_energies[favourite]

            simplified=True
            return short_circuit, resolver, energy, simplified
        else:
            simplified=False
            return gates_index, resolver, energy, simplified


    def simplify_circuit(self,indexed_circuit):
        """this function kills repeated unitaries and
        CNOTS and returns a simplified indexed_circuit vector"""
        #load circuit on each qubit
        connections={str(q):[] for q in range(self.n_qubits)} #this saves the gates in each qubit
        places_gates = {str(q):[] for q in range(self.n_qubits)} #this saves, for each gate on each qubit, the position in the original indexed_circuit


        flagged = [False]*len(indexed_circuit) #to check if you have seen a cnot already, so not to append it twice to the qubit's dictionary

        for q in range(self.n_qubits): #sweep over all qubits
            for nn,idq in enumerate(indexed_circuit): #sweep over all gates in original circuit's vector
                if idq<self.number_of_cnots: #if the gate it's a CNOT or not
                    control, target = self.indexed_cnots[str(idq)] #give control and target qubit
                    if q in [control, target] and not flagged[nn]: #if the qubit we are looking at is affected by this CNOT, and we haven't add this CNOT to the dictionary yet
                        connections[str(control)].append(idq)
                        connections[str(target)].append(idq)
                        places_gates[str(control)].append(nn)
                        places_gates[str(target)].append(nn)
                        flagged[nn] = True #so you don't add the other
                else:
                    if idq%self.n_qubits == q: #check if the unitary is applied to the qubit we are looking at
                        connections[str(q)].append("u")
                        places_gates[str(q)].append(nn)


        ### now reducing the circuit
        new_indexed_circuit = indexed_circuit.copy()
        for q, path in connections.items(): ###sweep over qubits: path is all the gates that act this qubit during the circuit
            for ind,gate in enumerate(path):
                if gate == "u": ## IF GATE IS SINGLE QUIT UNITARY, CHECK IF THE NEXT ONES ARE ALSO UNITARIES AND KILL 'EM
                    for k in range(len(path)-ind-1):
                        if path[ind+k+1]=="u":
                            new_indexed_circuit[places_gates[str(q)][ind+k+1]] = -1
                        else:
                            break
                elif gate in range(self.number_of_cnots) and ind<len(path)-1: ### self.number_of_cnots is the maximum index of a CNOT gate for a fixed self.n_qubits.
                    if path[ind+1]==gate and not (new_indexed_circuit[places_gates[str(q)][ind]] == -1): #check if the next gate is the same CNOT; and check if I haven't corrected the original one (otherwise you may simplify 3 CNOTs to id)
                        others = self.indexed_cnots[str(gate)].copy()
                        others.remove(int(q)) #the other qubit affected by the CNOT
                        for jind, jgate in enumerate(connections[str(others[0])][:-1]): ##sweep the other qubit's gates until i find "gate"
                            if jgate == gate and connections[str(others[0])][jind+1] == gate: ##i find the same gate that is repeated in both the original qubit and this one
                                if (places_gates[str(q)][ind] == places_gates[str(others[0])][jind]) and (places_gates[str(q)][ind+1] == places_gates[str(others[0])][jind+1]): #check that positions in the indexed_circuit are the same
                                 ###maybe I changed before, so I have repeated in the original but one was shut down..
                                    new_indexed_circuit[places_gates[str(q)][ind]] = -1 ###just kill the repeated CNOTS
                                    new_indexed_circuit[places_gates[str(q)][ind+1]] = -1 ###just kill the repeated CNOTS
                                    break

                if gate in range(self.number_of_cnots) and ind == 0: ###if I have a CNOT just before initializing, it does nothing (if |0> initialization).
                    others = self.indexed_cnots[str(gate)].copy()
                    others.remove(int(q)) #the other qubit affected by the CNOT
                    for jind, jgate in enumerate(connections[str(others[0])][:-1]): ##sweep the other qubit's gates until i find "gate"
                        if jgate == gate and jind==0: ##it's also the first gate in the other qubit
                            if (places_gates[str(q)][ind] == places_gates[str(others[0])][jind]): #check that positions in the indexed_circuit are the same
                                new_indexed_circuit[places_gates[str(q)][ind]] = -1 ###just kill the repeated CNOTS
                                break

        #### remove the marked indices ######
        #### remove the marked indices ######

        final=[]
        for gmarked in new_indexed_circuit:
            if not gmarked == -1:
                final.append(gmarked)
        return final

    def count_number_cnots(self, gates_index):
        c=0
        for k in gates_index:
            if k<self.number_of_cnots:
                c+=1
        return c
