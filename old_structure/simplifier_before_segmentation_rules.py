from circuit_basics import Basic
import numpy as np
import cirq
import sympy

class Simplifier(Basic):
    def __init__(self, n_qubits=3):
        """
        simplifies the circuit according to some rules that preserve the expected value of target hamiltornian.
        takes help from index_to_symbol (dict) and symbol_to_value (dict).
        Importantly, it keeps the parameter values of the untouched gates.

        It applies the following rules:
        Rules:  1. CNOT just after initializing, it does nothing (if |0> initialization).
                2. Two consecutive and equal CNOTS compile to identity.
                3. Rotation around z axis of |0> only adds phase hence leaves invariant <H>. Either kill it or replace it by Rx(0) (if we have no Rx in that wire, i.e. the ansatz becomes too simple, TFQ runs into problems).
                4. two equal rotations: add the values.
                5. Scan for U_3 = Rz Rx Rz, or Rx Rz Rx; if found, abosrb consecutive rz/rx (until a CNOT is found)
                6. Rz(control) and CNOT(control, target) Rz(control) --> Rz(control) CNOT
                7. Rx(target) and CNOT(control, target) Rx(target) --> Rx(target) CNOT

        Oiginally, it was a single big function now splitted into smaller ones..
        """
        super(Simplifier, self).__init__(n_qubits=n_qubits)
        self.single_qubit_unitaries = {"rx":cirq.rx, "rz":cirq.rz}

    def simplify_step(self, indexed_circuit, symbol_to_value, index_to_symbols):
        """
        Returns the (simplified) indexed_circuit, index_to_symbols, symbol_to_value.
        """
        connnections, places_gates = self.scan_qubits(indexed_circuit, index_to_symbols)
        new_indexed_circuit, NRE, symbols_on = self.simplify_intermediate(indexed_circuit, symbol_to_value, index_to_symbols, connnections, places_gates)
        Sindexed_circuit, Ssymbols_to_values, Sindex_to_symbols = self.translate_to_output(new_indexed_circuit, NRE, symbols_on)
        return Sindexed_circuit, Ssymbols_to_values, Sindex_to_symbols


    def check_qubits_on(self,circuit):
        """function that checks if all qubits are touched by a gate in the circuit"""
        check = True
        effective_qubits = list(circuit.all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                check = False
                break
        return check

    def compute_difference_states(self, indices1, resolver1, indices2, resolver2):
        up = np.zeros(2**self.n_qubits)
        up[0] = 1
        return np.dot(cirq.unitary(self.give_unitary(indices1,resolver1)) - cirq.unitary(self.give_unitary(indices2,resolver2)), up )


    def reduce_circuit(self, indexed_circuit, symbol_to_value,index_to_symbols, max_its=None):
        """
        iterate many times simplify circuit, break when no simplifying any more.
        """

        l0 = len(indexed_circuit)
        reducing = True

        if max_its is None:
            max_its = l0

        if self.check_qubits_on(self.give_circuit(indexed_circuit)[0]) is False:
            raise Error("Not all qubits being touched by a rotation! Please make your ansatz more complex.")

        for its in range(max_its):
            indexed_circuit, symbol_to_value,index_to_symbols = self.simplify_step(indexed_circuit, symbol_to_value,index_to_symbols)
            if len(indexed_circuit) == l0:
                break
        return indexed_circuit, symbol_to_value,index_to_symbols


    def rotation(self,vals):
        """
        Rz(\alpha) Rx(\beta) Rz(\gamma)
        with R_n(\theta) = \Exp[ -\ii \vec{theta} \vec{n} /2]
        """

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
                error=False
            except Exception:
                error=True
        return vals


    def scan_qubits(self,indexed_circuit, index_to_symbols):
        """
        this function scans the circuit as described by {indexed_circuit, index_to_symbols}
        and returns a dictionary with the gates acting on each qubit and the order of appearence on the original circuit
        """
        connections={str(q):[] for q in range(self.n_qubits)} #this saves the gates at each qubit. It does not respects the order.
        places_gates = {str(q):[] for q in range(self.n_qubits)} #this saves, for each gate on each qubit, the position in the original indexed_circuit
        flagged = [False]*len(indexed_circuit) #used to check if you have seen a cnot already, so not to append it twice to the qubit's dictionary

        for nn,idq in enumerate(indexed_circuit): #sweep over all gates in original circuit's list
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
                            connections[str(q)].append("rz")
                        elif self.n_qubits <= idq-self.number_of_cnots <  2*self.n_qubits:
                            connections[str(q)].append("rx")
                        places_gates[str(q)].append(nn)
                    flagged[nn] = True #to check that all gates have been flagged
        ####quick test
        for k in flagged:
            if k is False:
                raise Error("not all flags in flagged are True!")
        return connections, places_gates



    def simplify_intermediate(self,indexed_circuit, symbol_to_value,index_to_symbols, connections, places_gates):
        """
        Scans each qubit, and apply rules listed below to the circuit.

        Rules:  1. CNOT's control is |0>, kill that cnot.
                2. Two consecutive and equal CNOTS compile to identity.
                3. Rotation around z axis of |0> only adds phase hence leaves invariant <H>. Two options: kill it or replace it by Rx (to avoid having no gates)
                4. two equal rotations: add the values.
                5. Scan for U_3 = Rz Rx Rz, or Rx Rz Rx; if found, abosrb consecutive rz/rx (until a CNOT is found)
                6. Rz(control) and CNOT(control, target) Rz(control) --> Rz(control) CNOT
                7. Rx(target) and CNOT(control, target) Rx(target) --> Rx(target) CNOT
        """
        new_indexed_circuit = indexed_circuit.copy() #intermediate list of gates, the deleted ones are set to -1
        new_indexed_circuit_unitary = [False for k in indexed_circuit] #also used as a control if parametrized gates has been shut down
        symbols_to_delete=[] # list to store the symbols that will be deleted/modified
        symbols_on = {str(q):[] for q in list(connections.keys())}
        NRE ={} #NewREsolver

        for q, path in connections.items(): ###sweep over qubits: path is all the gates that act this qubit during the circuit
            for ind,gate in enumerate(path): ### for each qubit, sweep over the list of gates
                ##### CNOTS TIME ####

                if gate in range(self.number_of_cnots):
                    ### 1. if I have a CNOT just after initializing, it does nothing (if |0> initialization).
                    if ind == 0 and not new_indexed_circuit[places_gates[str(q)][ind]] == -1:
                        others = self.indexed_cnots[str(gate)].copy()
                        others.remove(int(q)) #the other qubit affected by the CNOT
                        control, target = self.indexed_cnots[str(indexed_circuit[places_gates[str(q)][ind]])]

                        for jind, jgate in enumerate(connections[str(others[0])]): ##Be sure it's the right gate
                            if (int(q) == control) and (jgate == gate) and (places_gates[str(q)][ind] == places_gates[str(others[0])][jind]):
                                new_indexed_circuit[places_gates[str(q)][ind]] = -1
                                break

                    elif ind<len(path)-1:
                        #2. 2 consecutive and equal CNOTS compile to identity.
                        next_gate_placed = new_indexed_circuit[places_gates[str(q)][ind+1]]
                        if not (gate_placed == -1 and next_gate_placed == -1):
                            if path[ind+1] == gate:
                                others = self.indexed_cnots[str(gate)].copy()
                                others.remove(int(q)) #the other qubit affected by the CNOT
                                for jind, jgate in enumerate(connections[str(others[0])][:-1]): ##sweep the other qubit's gates until i find "gate"
                                    if (jgate == gate) and (connections[str(others[0])][jind+1] == gate): ##i find the same gate that is repeated in both the original qubit and this one
                                        if (places_gates[str(q)][ind] == places_gates[str(others[0])][jind]) and (places_gates[str(q)][ind+1] == places_gates[str(others[0])][jind+1]): #check that positions in the indexed_circuit are the same
                                         ###maybe I changed before, so I have repeated in the original but one was shut down..
                                            new_indexed_circuit[places_gates[str(q)][ind]] = -1 ###just kill the repeated CNOTS
                                            new_indexed_circuit[places_gates[str(q)][ind+1]] = -1 ###just kill the repeated CNOTS
                                            break
                            else:
                                pass
                                #no symbols added.

                #### ROTATIONS ####
                elif gate in ["rz","rx"]:

                    gates = ["rz", "rx"] #which gate am I? Which gate are you?
                    gates.remove(gate) #which gate am I? Which gate are you?
                    other_gate = gates[0] #which gate am I? Which gate are you?

                    original_symbol = index_to_symbols[places_gates[str(q)][ind]]
                    original_value = symbol_to_value[original_symbol]

                    if ind==0: ### 3. Rotation around z axis of |0> does only add a phase, hence leaves invariant <H>. We kill it unless we get rid of rx.
                        if gate=="rz":
                            not_more_rxs = False
                            for ngs in path[ind+1:]:
                                if ngs in ["rx"]:
                                    symbols_to_delete.append(original_symbol)
                                    new_indexed_circuit[places_gates[str(q)][ind]] = -1
                                    not_more_rxs = True
                                    break
                            if not_more_rxs is False:
                                new_indexed_circuit[places_gates[str(q)][ind]] = self.number_of_cnots+self.n_qubits+int(q)
                                symbol_to_value[original_symbol] = 0
                                sname="th_"+str(len(list(NRE.keys())))
                                NRE[sname] = 0
                                symbols_on[str(q)].append(sname)
                        else:
                            sname="th_"+str(len(list(NRE.keys())))
                            NRE[sname] = original_value
                            symbols_on[str(q)].append(sname)
                    elif ind != len(path)-1:
                        ### 4. two equal rotations: add the values
                        next_gate_placed = new_indexed_circuit[places_gates[str(q)][ind+1]]
                        if path[ind+1] == gate and not next_gate_placed == -1:
                            next_symbol = index_to_symbols[places_gates[str(q)][ind+1]]
                            symbols_to_delete.append(next_symbol)
                            new_indexed_circuit[places_gates[str(q)][ind+1]] = -1

                            sname="th_"+str(len(list(NRE.keys())))
                            NRE[sname] = original_value + symbol_to_value[next_symbol]
                            symbols_on[str(q)].append(sname)
                            finish_here = True

                        elif ind< len(path)-2:
                            ## 5. Scan for U_3 = Rz Rx Rz, or Rx Rz Rx; if found, abosrb consecutive rz/rx (until a CNOT is found)
                            if path[ind+1] == other_gate and path[ind+2] == gate:
                                compile_gate = False
                                gate_to_compile = [self.single_qubit_unitaries[gate](original_value).on(self.qubits[int(q)])]

                                for pp in [1,2]: ##append next 2 gates to the gate_to_compile list (as appearing in the circuit)
                                    gate_to_compile.append(self.single_qubit_unitaries[path[ind+pp]](symbol_to_value[index_to_symbols[places_gates[str(q)][ind+pp]]]).on(self.qubits[int(q)]))

                                for ilum, next_gates_to_compile in enumerate(path[(ind+3):]): #Now scan the remaining part of that qubit line
                                    if next_gates_to_compile in ["rz","rx"] and not new_indexed_circuit[places_gates[str(q)][ind+3+ilum]] == -1:
                                        compile_gate = True #we'll compile!

                                        new_indexed_circuit[places_gates[str(q)][ind+3+ilum]] = -1
                                        dele = index_to_symbols[places_gates[str(q)][ind+3+ilum]]
                                        symbols_to_delete.append(dele)

                                        gate_to_compile.append(self.single_qubit_unitaries[next_gates_to_compile](symbol_to_value[dele]).on(self.qubits[int(q)]))
                                    else:
                                        break
                                if compile_gate: ### if conditions are met s.t. you can absorb everything into U_3:
                                    u = cirq.unitary(cirq.Circuit(gate_to_compile))
                                    vals = np.real(self.give_rz_rx_rz(u)[::-1]) #not entirely real since there's a finite number of iterations

                                    #### make sure this is rz rx rz
                                    new_indexed_circuit[places_gates[str(q)][ind]] = self.number_of_cnots+int(q)
                                    new_indexed_circuit[places_gates[str(q)][ind+1]] = self.number_of_cnots+int(q)+self.n_qubits
                                    new_indexed_circuit[places_gates[str(q)][ind+2]] = self.number_of_cnots+int(q)

                                    for o in range(3):
                                        new_indexed_circuit_unitary[places_gates[str(q)][ind+o]] = True

                                    for v in zip(list(vals)):
                                        sname="th_"+str(len(list(NRE.keys())))
                                        NRE[sname] = v[0]
                                        symbols_on[str(q)].append(sname)
                                    finish_here = True
                                else:
                                    if new_indexed_circuit_unitary[places_gates[str(q)][ind]] == False:
                                        sname="th_"+str(len(list(NRE.keys())))
                                        NRE[sname] = original_value
                                        symbols_on[str(q)].append(sname)
                            # 6. Rz(control) and CNOT(control, target) Rz(control) --> Rz(control) CNOT
                            elif (gate in ["rz","rx"]) and (path[ind+1] not in ["rx", "rz","u"]) and not new_indexed_circuit[places_gates[str(q)][ind+1]] == -1:
                                control, target = self.indexed_cnots[str(path[ind+1])]
                                values_to_add=[]
                                print(control, target)
                                if int(q) == control:
                                    if gate == "rz":
                                        ### scan for the next gates after CNOT and break when you find a new CNOT being target or Rx.
                                        for npip, pip in enumerate(path[ind+2:]):
                                            if (new_indexed_circuit[places_gates[str(q)][ind+2+npip]] == -1):
                                                break
                                            elif (pip not in ["rz"]):
                                                break
                                            elif not self.indexed_cnots[str(npip)][0] == int(q):
                                                break
                                            else:
                                                if (pip == "rz"):
                                                    next_symbol = index_to_symbols[places_gates[str(q)][ind+2+npip]]
                                                    symbols_to_delete.append(next_symbol)
                                                    new_indexed_circuit[places_gates[str(q)][ind+2+npip]] = -1
                                                    values_to_add.append(symbol_to_value[next_symbol])
                                                else:
                                                    break
                                        ### merge all the values into the first guy.
                                        if len(values_to_add)>0:
                                            sname="th_"+str(len(list(NRE.keys()))) ## this is safe, since we are looping on the indices first, and the resolver dict is ordered
                                            NRE[sname] = original_value + np.sum(values_to_add)
                                            symbols_on[str(q)].append(sname)
                                        else:
                                            if new_indexed_circuit_unitary[places_gates[str(q)][ind]] == False:
                                                sname="th_"+str(len(list(NRE.keys())))
                                                NRE[sname] = original_value
                                                symbols_on[str(q)].append(sname)
                                    else:
                                        if new_indexed_circuit_unitary[places_gates[str(q)][ind]] == False:
                                            sname="th_"+str(len(list(NRE.keys())))
                                            NRE[sname] = original_value
                                            symbols_on[str(q)].append(sname)

                                # 7. Rx(target) and CNOT(control, target) Rx(target) --> Rx(target) CNOT
                                elif int(q) == target:
                                    if gate == "rx":# and not :
                                        print("heeeere am i")
                                        for npip, pip in enumerate(path[ind+2:]):
                                            if (new_indexed_circuit[places_gates[str(q)][ind+2+npip]] == -1):
                                                break
                                            elif (pip not in ["rx"]):
                                                break
                                            elif not self.indexed_cnots[str(npip)][1] == int(q):
                                                break
                                            else:
                                                if (pip == "rx"):
                                                    next_symbol = index_to_symbols[places_gates[str(q)][ind+2+npip]]
                                                    symbols_to_delete.append(next_symbol)
                                                    new_indexed_circuit[places_gates[str(q)][ind+2+npip]] = -1
                                                    values_to_add.append(symbol_to_value[next_symbol])
                                                else:
                                                    break
                                        ### merge all the values into the first guy.
                                        if len(values_to_add)>0:
                                            sname="th_"+str(len(list(NRE.keys()))) ## this is safe, since we are looping on the indices first, and the resolver dict is ordered
                                            NRE[sname] = original_value + np.sum(value_to_add)
                                            symbols_on[str(q)].append(sname)
                                            finish_here = True
                                        else:
                                            if new_indexed_circuit_unitary[places_gates[str(q)][ind]] == False:
                                                sname="th_"+str(len(list(NRE.keys())))
                                                NRE[sname] = original_value
                                                symbols_on[str(q)].append(sname)
                                    else:
                                        if new_indexed_circuit_unitary[places_gates[str(q)][ind]] == False:
                                            sname="th_"+str(len(list(NRE.keys())))
                                            NRE[sname] = original_value
                                            symbols_on[str(q)].append(sname)

                                if new_indexed_circuit_unitary[places_gates[str(q)][ind]] == False:
                                    sname="th_"+str(len(list(NRE.keys())))
                                    NRE[sname] = original_value
                                    symbols_on[str(q)].append(sname)
                            else:
                                if new_indexed_circuit_unitary[places_gates[str(q)][ind]] == False:
                                    sname="th_"+str(len(list(NRE.keys())))
                                    NRE[sname] = original_value
                                    symbols_on[str(q)].append(sname)

                        else:
                            if new_indexed_circuit_unitary[places_gates[str(q)][ind]] == False:
                                sname="th_"+str(len(list(NRE.keys())))
                                NRE[sname] = original_value
                                symbols_on[str(q)].append(sname)

                    else: #if nothing to change, just add the 1-qbit gate as it is.
                        if new_indexed_circuit_unitary[places_gates[str(q)][ind]] == False:
                            sname="th_"+str(len(list(NRE.keys())))
                            NRE[sname] = original_value
                            symbols_on[str(q)].append(sname)


        return new_indexed_circuit, NRE, symbols_on

    def translate_to_output(self, new_indexed_circuit, NRE, symbols_on):
        """
        Final building block for the simplifying function.

        this function takes a processed new_indexed_circuit (with marked gates to deleted to be -1)
        a new resolver NRE with symbols whose corresponding qubit is obtained from symbols_on dict. Symbols_dict contains, for each qubit, the list of symbols acting on it, apppearing in order, when sweeping on the original indexed_circuit. The current function sweeps again over the original indexed_circuit, and marks for each qubit the symbols already used.

        """
        final=[]
        final_idx_to_symbols={}
        final_dict = {}

        index_gate=0
        for gmarked in new_indexed_circuit:
            if not gmarked == -1:
                final.append(gmarked)
                if 0 <= gmarked - self.number_of_cnots < 2*self.n_qubits:
                    ### in which position we add this symbol ?
                    for indd, k in enumerate(symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)]):
                        if k != -1:
                            break
                    final_idx_to_symbols[int(len(final)-1)] = "th_"+str(len(list(final_dict.keys())))
                    final_dict["th_"+str(len(list(final_dict.keys())))] = NRE[symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd]]
                    symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd]=-1
                else:
                    final_idx_to_symbols[int(len(final)-1)] = ""

        return final, final_dict, final_idx_to_symbols




# the same (and equal to zero)#
    #rx_next = False
    #for ngs in path[ind+1:]:
#        if ngs in ["rx"]:___#
#        rx_next = True
#        break
#### if you run out of gates: add Rx at both control and target
#if rx_next is False:#
#    new_indexed_circuit[places_gates[str(q)][ind]] = self.number_of_cnots+self.n_qubits+int(q)#
#    new_indexed_circuit[places_gates[str(others[0])][jind]] = self.number_of_cnots+self.n_qubits+int(others[0])
    #for qub in [q, others[0]]:
#        sname="th_"+str(len(list(NRE.keys())))
#        NRE[sname] = 0
#        symbols_on[str(qub)].append(sname)
#break
