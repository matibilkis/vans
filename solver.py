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
from datetime import datetime


class Solver:
    def __init__(self, n_qubits=3, qlr=0.01, qepochs=10**4,verbose=0, g=1, J=0, noise=False, noise_level=0.01, patience=100):

        """
        patience: used at EarlyStopping in training
        """
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.lower_bound_Eg = -2*self.n_qubits

        self.qlr = qlr
        self.qepochs=qepochs
        self.verbose=verbose

        self.patience = patience

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
        self.single_qubit_unitaries = {"rx":cirq.rx, "rz":cirq.rz}

        self.observable=self.ising_obs(g=g, J=J)

        self.noise = noise
        self.noise_level = noise_level

    def ising_obs(self, g=1, J=0):
        self.g=g
        self.J=J
        observable = [-float(0.5*g)*cirq.Z.on(q) for q in self.qubits]
        for q in range(len(self.qubits)):
            observable.append(-float(0.5*J)*cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
        return observable

    def append_to_circuit(self, ind, circuit, params, index_to_symbols):
        #### add CNOT
        if ind < self.number_of_cnots:
            control, target = self.indexed_cnots[str(ind)]
            circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            if isinstance(index_to_symbols,dict):
                index_to_symbols[len(list(index_to_symbols.keys()))] = []
                return circuit, params, index_to_symbols
            else:
                return circuit, params

        #### add rz #####
        elif 0 <= ind - self.number_of_cnots  < self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(1),[cirq.rz]):
                new_param = "th_"+str(len(params))
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_param
                return circuit, params, index_to_symbols

        #### add rx #####
        elif self.n_qubits <= ind - self.number_of_cnots  < 2*self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(1),[cirq.rx]):
                new_param = "th_"+str(len(params))
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_param
            return circuit, params, index_to_symbols

    def give_unitary(self,idx, res):
        return cirq.resolve_parameters(self.give_circuit(idx)[0], res)

    def give_circuit(self, lista):
        circuit, symbols, index_to_symbols = [], [], {}
        for k in lista:
            circuit, symbols, index_to_symbols = self.append_to_circuit(k,circuit,symbols, index_to_symbols)
        circuit = cirq.Circuit(circuit)
        return circuit, symbols, index_to_symbols



    def simplify_kill_simplify(self, indexed_circuit, index_to_symbols, symbol_to_value, energy_bound , max_its=None):
        """
        given a circuit configuration, it reduces (iterates many times simplifier function), then try to kill one unitary and then simplify again.
        """
        indexed_circuit, index_to_symbols, symbol_to_value = self.reduce_circuit(indexed_circuit, index_to_symbols, symbol_to_value)

        l0 = len(indexed_circuit)
        if max_its is None:
            max_its = l0

        simplified, itt = True, 0
        while simplified or itt<max_its:
            itt+=1
            simplified, _ = self.kill_one_unitary(indexed_circuit, index_to_symbols, symbol_to_value)
            if simplified:
                indexed_circuit, index_to_symbols, symbol_to_value = _
                indexed_circuit, index_to_symbols, symbol_to_value = self.reduce_circuit(indexed_circuit, index_to_symbols, symbol_to_value)
            else:
                break
        return indexed_circuit, index_to_symbols, symbol_to_value




def diff(u_1, u_2, cnots_simplified = False, numpy_type=True):
    ui = cirq.unitary(u_1)
    uf = cirq.unitary(u_2)
    if cnots_simplified:
        return np.sum(np.abs((ui - uf)[:,0]))
    else:
        return np.sum(np.abs((ui - uf)))

def diff_expectation(u1,u2,whole=False):
    e=[]
    for u in [u1, u2]:
        effective_qubits = list(u.all_qubits())
        for k in sol.qubits:
            if k not in effective_qubits:
                u.append(cirq.I.on(k))
        expectation_layer = tfq.layers.Expectation()
        tfqciru1 = tfq.convert_to_tensor([u]) ###SymbolToValue parameters !!!
        exp1 = expectation_layer(tfqciru1,
                                    operators=tfq.convert_to_tensor([sol.observable]))
        e.append(np.float32(np.squeeze(tf.math.reduce_sum(exp1, axis=-1, keepdims=True))))
    if whole:
        return e
    return e[0] - e[1]

def plot(data):
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot2grid((1,1),(0,0))
    ax.scatter(range(len(data)),data, c="red", alpha=.5)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")

    return fig
