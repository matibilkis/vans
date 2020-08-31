import gc
import numpy as np
import sympy
import cirq
import tensorflow_quantum as tfq
from tqdm import tqdm
import tensorflow as tf
import argparse
import os
import pickle
import matplotlib.pyplot as plt




class GeneticSolver:
    def __init__(self, n_qubits=3, qlr=0.01, qepochs=100,verbose=0, g=1, J=0, noise=False, noise_level=0.01):

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


        self.observable=self.ising_obs(g=g, J=J)
        self.single_qubit_unitaries = {"u":self.parametrized_unitary, "rx":[cirq.rx], "rz":[cirq.rz]}

        self.noise = noise
        self.noise_level = noise_level

    def ising_obs(self, g=1, J=0):
        self.g=g
        self.J=J
        observable = [-float(0.5*g)*cirq.Z.on(q) for q in self.qubits]
        for q in range(len(self.qubits)):
            observable.append(-float(0.5*J)*cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
        return observable

    def append_to_circuit(self, ind, circuit, params, index_to_symbols=None):
        #### add CNOT
        if ind < self.number_of_cnots:
            control, target = self.indexed_cnots[str(ind)]
            circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            if isinstance(index_to_symbols,dict):
                index_to_symbols[len(list(index_to_symbols.keys()))] = []
                return circuit, params, index_to_symbols
            else:
                return circuit, params

        ### add parametrized_unitary ####
        elif self.number_of_cnots <= ind  < self.number_of_cnots+self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            new_params=[]
            for par, gate in zip(range(3),self.parametrized_unitary):
                new_param = "th_"+str(len(params))
                new_params.append(new_param)
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
            if isinstance(index_to_symbols,dict):
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_params
                return circuit, params, index_to_symbols
            else:
                return circuit, params

        #### add rz #####
        elif self.n_qubits <= ind - self.number_of_cnots  < 2*self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            new_params=[]
            for par, gate in zip(range(1),[cirq.rz]):
                new_param = "th_"+str(len(params))
                new_params.append(new_param)
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
            if isinstance(index_to_symbols,dict):
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_params
                return circuit, params, index_to_symbols
            else:
                return circuit, params

        #### add rx #####
        elif 2*self.n_qubits <= ind - self.number_of_cnots  < 3*self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            new_params=[]
            for par, gate in zip(range(1),[cirq.rx]):
                new_param = "th_"+str(len(params))
                new_params.append(new_param)
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))

            if isinstance(index_to_symbols,dict):
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_params
                return circuit, params, index_to_symbols
            else:
                return circuit, params

    def give_unitary(self,idx, res):
        return cirq.resolve_parameters(self.give_circuit(idx)[0], res)

    def give_circuit(self, lista,give_index_to_symbols=True):
        if give_index_to_symbols:
            circuit, symbols, index_to_symbols = [], [], {}
            for k in lista:
                circuit, symbols, index_to_symbols = self.append_to_circuit(k,circuit,symbols, index_to_symbols)
            circuit = cirq.Circuit(circuit)
            return circuit, symbols, index_to_symbols
        else:
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


    def unravel(self, gates_index):#, ordered_symbol_to_value):
        """
        this function gives the circuit in terms of single-qubit untaries explicitely (without the index associated to u)

        assuming the symbol corresponds to gates_index (if I have number_of_cnots < index < Ncnots + n_qubits)
        it means we've rz(\theta_0+len(symbols_appended_so_far)) rx(\theta_1+len(symbols_appended_so_far)
        rz("th_2+len(symbols_appended_so_far)).
        """
        unraveled = []
        for nn, idq in enumerate(gates_index):
            if 0<= idq - self.number_of_cnots  < self.n_qubits:
                #target_qubit = (idq - self.number_of_cnots)%self.n_qubits
                for addition in [1,2,1]:
                    unraveled.append(idq+(self.n_qubits*addition))
            else:
                unraveled.append(idq)
        return unraveled


    def TFQ_model(self, symbols, lr):
        circuit_input = tf.keras.Input(shape=(), dtype=tf.string)

        if self.noise is True:
            output = tfq.layers.Expectation(backend=cirq.DensityMatrixSimulator(noise=cirq.depolarize(self.noise_level)))(
                    circuit_input,
                    symbol_names=symbols,
                    operators=tfq.convert_to_tensor([self.observable]),
                    initializer=tf.keras.initializers.RandomNormal(stddev=np.pi/2))

        else:
            output = tfq.layers.Expectation()(
                    circuit_input,
                    symbol_names=symbols,
                    operators=tfq.convert_to_tensor([self.observable]),
                    initializer=tf.keras.initializers.RandomNormal(stddev=np.pi/2))

        model = tf.keras.Model(inputs=circuit_input, outputs=output)
        adam = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=adam, loss='mse')
        return model

    def initialize_model_insertion(self, symbols_to_values):
        ## notice I add some noise! maybe we want to vary this noise according to how "stucked" the algorithm is
        model = self.TFQ_model(list(symbols_to_values.keys()), lr=self.qlr)
        model.trainable_variables[0].assign(
            tf.convert_to_tensor(np.array(list(symbols_to_values.values())).astype(np.float32))) #initialize parameters of model (continuous parameters of unitaries) + 0.1*np.random.randn(len(list(symbols_to_values.values()))).astype(np.float32)
        return model



    def compute_energy_first_time(self, circuit, symbols, hyperparameters):
        """
        takes as input vector with actions described as integer
        and outputsthe energy of that circuit (w.r.t self.observable)

        hyperparameters = [epoch, lr]
        """

        ### this is because each qubit should be "activated" in TFQ to do the optimization (if the observable has support on this qubit as well and you don't add I then error)
        effective_qubits = list(circuit.all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                circuit.append(cirq.I.on(k))

        tfqcircuit = tfq.convert_to_tensor([circuit])

        model = self.TFQ_model(symbols, hyperparameters[1])
        qoutput = tf.ones((1, 1))*self.lower_bound_Eg ##### important to discuss with Lucksasz.
        #print("about to fit!")
        h=model.fit(x=tfqcircuit, y=qoutput, batch_size=1,
                    epochs=hyperparameters[0], verbose=self.verbose, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode="min")])
        energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))
        final_params = model.trainable_variables[0].numpy()
        resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
        del model
        gc.collect()

        return resolver, energy,h



    def rotation(self,vals):
        alpha,beta,gamma = vals
        return np.array([[np.cos(beta/2)*np.cos(alpha/2 + gamma/2) - 1j*np.cos(beta/2)*np.sin(alpha/2 + gamma/2),
                 (-1j)*np.cos(alpha/2 - gamma/2)*np.sin(beta/2) - np.sin(beta/2)*np.sin(alpha/2 - gamma/2)],
                [(-1j)*np.cos(alpha/2 - gamma/2)*np.sin(beta/2) + np.sin(beta/2)*np.sin(alpha/2 - gamma/2),
                 np.cos(beta/2)*np.cos(alpha/2 + gamma/2) + 1j*np.cos(beta/2)*np.sin(alpha/2 + gamma/2)]])


    def give_rz_rx_rz(self,u):
        """
        finds \alpha, \beta \gamma s.t m = Rz(\alpha) Rx(\beta) Rz(\gamma)
        ****
        input: 2x2 unitary matrix as numpy array
        output: [\alpha \beta \gamma]
        """
        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        g = sympy.Symbol("g")

        eqs = [sympy.exp(-sympy.I*.5*(a+g))*sympy.cos(.5*b) ,
               -sympy.I*sympy.exp(-sympy.I*.5*(a-g))*sympy.sin(.5*b),
                sympy.exp(sympy.I*.5*(a+g))*sympy.cos(.5*b)
              ]

        kk = np.reshape(u, (4,))
        s=[]
        for i,r in enumerate(kk):
            if i!=2:
                s.append(r)

        t=[]
        for eq, val in zip(eqs,s):
            t.append((eq)-np.round(val,5))

        ### this while appears since the seed values may enter in vanishing gradients and through Matrix-zero error.
        error=True
        while error:
            try:
                solution = sympy.nsolve(t,[a,b,g],np.pi*np.array([np.random.random(),np.random.random(),np.random.random()]) ,maxsteps=3000, verify=True)
                vals = np.array(solution.values()).astype(np.complex64)
                #print(np.round(rotation(vals),3)-m)
                error=False
            except Exception:
                error=True
        return vals

    def prepare_circuit_insertion(self, gates_index, block_to_insert, insertion_index, SymbolsToValues,ep=0.01, init_params="PosNeg"):
        """gates_index is a vector with integer entries, each one describing a gate
            block_to_insert is block of unitaries to insert at index insertion
        """
        idx_circuit=[]
        symbols = []
        new_symbols = []
        new_resolver = {}
        full_resolver={} #this is the final output
        full_indices=[] #this is the final output

        if gates_index == []:
            indices = [-1]
        else:
            indices = gates_index
        for ind, g in enumerate(indices):
            #### insert new block ####
            if ind == insertion_index:
                for gate in block_to_insert:
                    full_indices.append(gate)

                    idx_circuit.append(gate)
                    if gate < self.number_of_cnots:
                        control, target = self.indexed_cnots[str(gate)]
                    else: ### i do only add block of unitaries.
                        qubit = self.qubits[(gate-self.number_of_cnots)%self.n_qubits]
                        ### with prob .5 I add
                        if init_params == "PosNeg":
                            rot = np.random.uniform(-np.pi,np.pi)
                            new_values = [rot, np.random.uniform(-ep,ep), -rot]
                        else:
                            new_values = [np.random.uniform(-ep,ep) for oo in range(3)]

                        for par, gateblack in zip(range(3),self.parametrized_unitary):
                            new_symbol = "New_th_"+str(len(new_symbols))
                            new_symbols.append(new_symbol)
                            new_resolver[new_symbol] = new_values[par] #rotation around epsilon... we can do it better afterwards
                            full_resolver["th_"+str(len(full_resolver.keys()))] = new_resolver[new_symbol]
            if 0<= g < self.number_of_cnots:
                full_indices.append(g)
                idx_circuit.append(g)
                control, target = self.indexed_cnots[str(g)]
            elif g>= self.number_of_cnots:
                full_indices.append(g)
                idx_circuit.append(g)
                qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]

                if 0<= g-self.number_of_cnots < self.n_qubits:
                    for par, gate in zip(range(3),self.parametrized_unitary):
                        new_symbol = "th_"+str(len(symbols))
                        symbols.append(new_symbol)
                        full_resolver["th_"+str(len(full_resolver.keys()))] = SymbolsToValues[new_symbol]


                if self.n_qubits<= g-self.number_of_cnots < 2*self.n_qubits:
                    new_symbol = "th_"+str(len(symbols))
                    symbols.append(new_symbol)
                    full_resolver["th_"+str(len(full_resolver.keys()))] = SymbolsToValues[new_symbol]

                if 2*self.n_qubits<= g-self.number_of_cnots < 3*self.n_qubits:
                    new_symbol = "th_"+str(len(symbols))
                    symbols.append(new_symbol)
                    full_resolver["th_"+str(len(full_resolver.keys()))] = SymbolsToValues[new_symbol]


        _,_, IndexToSymbols = self.give_circuit(full_indices)

        return full_indices, IndexToSymbols, full_resolver


    def optimize_model_from_indices(self, indices, model):
        circuit, variables, _ = self.give_circuit(indices)
        effective_qubits = list(circuit.all_qubits())

        for k in self.qubits:#che, lo que no estoy
            if k not in effective_qubits:
                circuit.append(cirq.I.on(k))

        tfqcircuit = tfq.convert_to_tensor([circuit])

        qoutput = tf.ones((1, 1))*self.lower_bound_Eg
        h=model.fit(x=tfqcircuit, y=qoutput, batch_size=1, epochs=self.qepochs,
                  verbose=self.verbose, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode="min")])
        energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))
        return energy,h

    def check_qubits_on(self,circuit):
        check = True
        effective_qubits = list(circuit.all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                check = False
                break
        return check

    def reduce_circuit(self, simp_indices, idxToSymbol, SymbolToVal, max_its=None):
        l0 = len(simp_indices)
        reducing = True
        if max_its is None:
            max_its = l0

        for its in range(max_its):
            simp_indices, idxToSymbol, SymbolToVal = self.simplify_circuit(simp_indices, idxToSymbol, SymbolToVal) #it would be great to have a way to realize that insertion was trivial...RL? :-)
            if len(simp_indices) == l0 or self.check_qubits_on(self.give_circuit(simp_indices)[0]) is False:
                reducing = False

        return simp_indices, idxToSymbol, SymbolToVal


    def simplify_circuit(self,indexed_circuit, index_to_symbols, symbol_to_value):#, symbol_to_position):
        """
        what i do:

        1) if you have rz, rx, rz then kill all future 1-qubit unitaries (if not cnot) on that qubit
        2) if you have a cnot as first gate, kill it
        3) two cnots compile to identity

        things to do:
        1) if you have rz in the beggining, kill it (but this can be done with the kill_on_unitary!)
        2) if you have rx after target, move it if you can simplify the circuit further (this is more subtle, maybe i should unravel)

        """
        #load circuit on each qubit
        #position_to_symbol = {str(k):l for k,l in zip(symbol_to_position.values(), symbol_to_position.keys())}

        connections={str(q):[] for q in range(self.n_qubits)} #this saves the gates in each qubit. Notice that this does not necessarily respects the order.
        places_gates = {str(q):[] for q in range(self.n_qubits)} #this saves, for each gate on each qubit, the position in the original indexed_circuit

        flagged = [False]*len(indexed_circuit) #to check if you have seen a cnot already, so not to append it twice to the qubit's dictionary

        for nn,idq in enumerate(indexed_circuit): #sweep over all gates in original circuit's vector
            for q in range(self.n_qubits): #sweep over all qubits
                if idq<self.number_of_cnots: #if the gate it's a CNOT or not
                    control, target = self.indexed_cnots[str(idq)] #give control and target qubit
                    if q in [control, target] and not flagged[nn]: #if the qubit we are looking at is affected by this CNOT, and we haven't add this CNOT to the dictionary yet
                        connections[str(control)].append(idq)
                        connections[str(target)].append(idq)
                        places_gates[str(control)].append(nn)
                        places_gates[str(target)].append(nn)
                        flagged[nn] = True #so you don't add the other
                else:
                    if (idq-self.number_of_cnots)%self.n_qubits == q: #check if the unitary is applied to the qubit we are looking at
                        if 0 <= idq - self.number_of_cnots< self.n_qubits:
                            connections[str(q)].append("u")
                        elif self.n_qubits <= idq-self.number_of_cnots <  2*self.n_qubits:
                            connections[str(q)].append("rz")
                        elif 2*self.n_qubits <= idq-self.number_of_cnots <  3*self.n_qubits:
                            connections[str(q)].append("rx")
                        places_gates[str(q)].append(nn)
                    flagged[nn] = True #to check that all gates have been flagged

        #### testing ###
        err=False
        for k in flagged:
            if k is not True:
                err = True
        if err is True:
            raise Error("not all gates have been flagged! check Solver.simplify method")

        ### now reducing the circuit
        new_indexed_circuit = indexed_circuit.copy()
        new_symbol_to_value = symbol_to_value.copy()
        flagged_symbols = {k:True for k in list(symbol_to_value.keys())}

        NRE ={}
        symbols_to_delete=[]
        symbols_to_use = []
        symbols_on = {str(q):[] for q in list(connections.keys())}
        for q, path in connections.items(): ###sweep over qubits: path is all the gates that act this qubit during the circuit
            for ind,gate in enumerate(path):
                if gate == "u" and not new_indexed_circuit[places_gates[str(q)][ind]] == -1: ## IF GATE IS SINGLE QUIT UNITARY, CHECK IF THE NEXT ONES ARE ALSO UNITARIES AND KILL 'EM
                    gate_to_compile=[]
                    pars_here=[]
                    compile_gate=False
                    symbols_deleted_here=[]
                    for ug, symbol in zip(self.single_qubit_unitaries[gate], index_to_symbols[places_gates[str(q)][ind]]):
                        value_symbol = symbol_to_value[symbol]
                        gate_to_compile.append(ug(value_symbol).on(self.qubits[int(q)]))
                        pars_here.append(symbol)

                    for k in range(len(path)-ind-1):
                        if path[ind+k+1] in list(self.single_qubit_unitaries.keys()):
                            new_indexed_circuit[places_gates[str(q)][ind+k+1]] = -1
                            compile_gate = True #we'll compile!
                            for ug, symbol in zip(self.single_qubit_unitaries[gate], index_to_symbols[places_gates[str(q)][ind+k+1]]):
                                value_symbol = symbol_to_value[symbol]
                                symbols_to_delete.append(symbol)
                                symbols_deleted_here.append(symbol)
                                gate_to_compile.append(ug(value_symbol).on(self.qubits[int(q)]))
                        else:
                            break
                    if compile_gate:
                        u = cirq.unitary(cirq.Circuit(gate_to_compile))
                        vals = np.real(self.give_rz_rx_rz(u)[::-1]) #not entirely real since finite number of iterations
                        for smb,v in zip(pars_here,vals):
                            sname = "th_"+str(int(smb.replace("th_",""))+len(symbols_to_delete)-len(symbols_deleted_here))
                            NRE[sname] = v
                            symbols_on[str(q)].append(sname)
                    else:
                        old_values = [symbol_to_value[sym] for sym in index_to_symbols[places_gates[str(q)][ind]]]

                        for smb,v in zip(pars_here,old_values):
                            #print(smb)
                            ##### this +len(symbols_to_delete) should respect qubit order..
                            sname="th_"+str(int(smb.replace("th_",""))+len(symbols_to_delete)-len(symbols_deleted_here))
                            NRE[sname] = v
                            symbols_on[str(q)].append(sname)

                if gate in range(self.number_of_cnots) and ind<len(path)-1: ### self.number_of_cnots is the maximum index of a CNOT gate for a fixed self.n_qubits.
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


        final=[]
        final_values={}
        fpars=[]
        final_idx_to_symbols={}
        final_dict = {}

        SN=0
        for gmarked in new_indexed_circuit:
            if not gmarked == -1:
                final.append(gmarked)
                final_idx_to_symbols[int(len(final)-1)] = []

                if 0 <= gmarked - self.number_of_cnots < self.n_qubits:
                    for indd, sym in enumerate(symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)]):
                        if sym != -1:
                            break
                    for k in range(3):
                        final_idx_to_symbols[int(len(final)-1)].append("th_"+str(SN))#symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd+k])
                        final_dict["th_"+str(SN)] = NRE[symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd+k]]
                        symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd+k]=-1
                        SN+=1
                if self.n_qubits <= gmarked - self.number_of_cnots < 3*self.n_qubits:
                    for indd, sym in enumerate(symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)]):
                        if sym != -1:
                            break
                    for k in range(1):
                        final_idx_to_symbols[int(len(final)-1)].append("th_"+str(SN))#symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd+k])
                        final_dict["th_"+str(SN)] = NRE[symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd+k]]
                        symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd+k]=-1
                        SN+=1

        #_,_, final_idx_to_symbols = self.give_circuit(final)
        return final, final_idx_to_symbols, final_dict



    def kill_one_unitary(self, gates_index, SymbolToValue, historial, mode="input_gate"):
        """notice this accepts either gates_index that encode u = [rz rx rz] or single gates (rz, rx).
            Importantly SymbolToValue respects the order of gates_index.
        """

        circuit_proposals=[] #storing all good candidates.
        circuit_proposals_energies=[]

        if self.noise is True:
            expectation_layer = tfq.layers.Expectation(backend=cirq.DensityMatrixSimulator(noise=cirq.depolarize(self.noise_level)))

        else:
            expectation_layer = tfq.layers.Expectation()


        tfqcircuit_gates_index_energy = tfq.convert_to_tensor([cirq.resolve_parameters(self.give_circuit(gates_index)[0], SymbolToValue)]) ###SymbolToValue parameters !!!
        #backend=cirq.DensityMatrixSimulator(noise=cirq.depolarize(self.noise_level))
        expval_gates_index = expectation_layer(  tfqcircuit_gates_index_energy,
                                operators=tfq.convert_to_tensor([self.observable]))
        gates_index_energy = np.float32(np.squeeze(tf.math.reduce_sum(expval_gates_index, axis=-1, keepdims=True)))

        if not mode=="input_gate":
            compare_energy=historial.lowest_energy
        else:
            compare_energy = gates_index_energy

        for i1, j in enumerate(gates_index):
            indexed_prop=[]

            prop=cirq.Circuit()
            checking = False
            ko=0
            to_pop=[]

            for i2, k in enumerate(gates_index):
                if k < self.number_of_cnots:
                    indexed_prop.append(k)
                    control, target = self.indexed_cnots[str(k)]
                    prop.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
                else:
                    if i1 != i2:
                        indexed_prop.append(k)
                        qubit = self.qubits[(k-self.number_of_cnots)%self.n_qubits]
                        if 0<= k-self.number_of_cnots < self.n_qubits:
                            for par, gate in zip(range(3),self.parametrized_unitary):
                                new_param = "th_"+str(ko)
                                ko+=1
                                prop.append(gate(sympy.Symbol(new_param)).on(qubit))
                        else:
                            if self.n_qubits <= k-self.number_of_cnots < 2*self.n_qubits:
                                new_param = "th_"+str(ko)
                                ko+=1
                                prop.append(cirq.rz(sympy.Symbol(new_param)).on(qubit))

                            if 2*self.n_qubits <= k-self.number_of_cnots < 3*self.n_qubits:
                                new_param = "th_"+str(ko)
                                ko+=1
                                prop.append(cirq.rx(sympy.Symbol(new_param)).on(qubit))

                    else: #i1 == i2
                        checking=True
                        if 0<= k-self.number_of_cnots < self.n_qubits:
                            for i in range(3):
                                to_pop.append("th_"+str(ko))
                                ko+=1
                        else:
                            if self.n_qubits <= k-self.number_of_cnots < 3*self.n_qubits:
                                to_pop.append("th_"+str(ko))
                                ko+=1


            if checking is True:
                reject = False
                nr = SymbolToValue.copy()
                for p in to_pop:
                    nr.pop(p)

                #### check if I actually kill a part of the circuit (for sure this won't help)
                connections={str(q):[] for q in range(self.n_qubits)} #this saves the gates in each qubit. Notice that this does not necessarily respects the order.

                for nn,idq in enumerate(indexed_prop): #sweep over all gates in original circuit's vector
                    for q in range(self.n_qubits): #sweep over all qubits
                        if (idq-self.number_of_cnots)%self.n_qubits == q and (idq>=self.number_of_cnots): #check if the unitary is applied to the qubit we are looking at
                            connections[str(q)].append("hey gate")

                for q in range(self.n_qubits): #sweep over all qubits
                    if connections[str(q)] == []:
                        reject = True
                        break

                if reject is False:
                    tfqcircuit = tfq.convert_to_tensor([cirq.resolve_parameters(prop, nr)]) ###SymbolToValue parameters !!!
                    #backend=cirq.DensityMatrixSimulator(noise=cirq.depolarize(self.noise_level))
                    expval = expectation_layer(  tfqcircuit,
                                            operators=tfq.convert_to_tensor([self.observable]))
                    new_energy = np.float32(np.squeeze(tf.math.reduce_sum(expval, axis=-1, keepdims=True)))
                    #if historial.accept_energy(new_energy, kill_one_unitary=True):
                    #if (new_energy - gates_index_energy)/np.abs(gates_index_energy) <= 0.01:

                    if (new_energy - compare_energy)/np.abs(compare_energy) <= 0.01:

                    # if new_energy < gatex_index even better.
                        #if new_energy > gates_index ---> new_energy - gatex_index > 0
                        # let's allow that at most new_energy = gatex_index_energy + 0.01*np.abs(gates_index_energy)
                        # hence we want that new_energy < gates_index_enery + 0.01*np.abs(gates_index_enery)

                        ordered_SymbolToValue = {}
                        for ind,k in enumerate(nr.values()):
                            ordered_SymbolToValue["th_"+str(ind)] = k
                        circuit_proposals.append([indexed_prop,ordered_SymbolToValue,new_energy])
                        circuit_proposals_energies.append(new_energy)

        del expectation_layer
        if len(circuit_proposals)>0:
            favourite = np.where(np.array(circuit_proposals_energies) == np.min(circuit_proposals_energies))[0][0]
            short_circuit, SymbolToValue, energy = circuit_proposals[favourite]
            simplified=True
            _,_, simplified_index_to_symbols = self.give_circuit(short_circuit)
            return short_circuit, SymbolToValue, energy, simplified_index_to_symbols, simplified, gates_index_energy
        else:
            simplified=False
            return gates_index, SymbolToValue, None, None, simplified,gates_index_energy



    def simplify_kill_simplify(self, NewIndices, New_Idx_To_Symbols,New_SymbolToValue, historial , max_its=None):
        simp_indices, idxToSymbol, SymbolToVal = self.reduce_circuit(NewIndices, New_Idx_To_Symbols,New_SymbolToValue)

        l0 = len(simp_indices)
        if max_its is None:
            max_its = l0

        simplified, itt = True, 0
        while simplified or itt<max_its:
            itt+=1
            simp_indices, SymbolToVal, new_energy, idxToSymbol, simplified, _ = self.kill_one_unitary(simp_indices, SymbolToVal, historial)
            if simplified:
                ### iterate many times circuit simplification
                simp_indices, idxToSymbol, SymbolToVal = self.reduce_circuit(simp_indices, idxToSymbol, SymbolToVal)
            else:
                break
        return simp_indices, idxToSymbol, SymbolToVal
