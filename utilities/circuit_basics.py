import cirq
import numpy as np
import sympy
import pickle
import os
from datetime import datetime
from glob import glob
from functools import wraps
import errno
import os
import signal


class Basic:
    def __init__(self, n_qubits=3, testing=False):
        """
        n_qubits: number of qubits on your ansatz
        testing: this is inherited by other classes to ease the debugging.
        """
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.lower_bound_Eg = -20*self.n_qubits

        #### keep a register on which integers corresponds to which CNOTS, target or control.
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

        self.testing=testing


    def append_to_circuit(self, ind, circuit, params, index_to_symbols):
        """
        ind: integer describing the gate to append to circuit
        circuit: cirq object that describes the quantum circuit
        params: a list containing the symbols appearing in the circuit so far
        index_to_sybols: tells which symbol corresponds to i^{th} item in the circuit (useful for simplifier)
        """

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

    def give_qubit(self, ind):
        """
        returns a list of qubits affected by gate indexed via ind
        used for cirq.insert_batch in the noise
        """
        if ind < self.number_of_cnots:
            return self.indexed_cnots[str(ind)]
        else:
            return [(ind-self.number_of_cnots)%self.n_qubits]

    def give_circuit(self, lista):
        """
        retrieves circuit (cirq object), with a list of each continuous parameter (symbols) and dictionary index_to_symbols giving the symbols for each position (useful for simplifier)

        lista: list of integers in [0, 2*n + n(n-1)), with n = self.number_qubits. Each integer describes a possible unitary (CNOT, rx, rz)
        """
        circuit, symbols, index_to_symbols = [], [], {}
        for k in lista:
            circuit, symbols, index_to_symbols = self.append_to_circuit(k,circuit,symbols, index_to_symbols)
        circuit = cirq.Circuit(circuit)
        return circuit, symbols, index_to_symbols

    def give_unitary(self,idx, res):
        """
        a shortcut to resolve parameters.

        idx: sequence of integers encoding the gates
        res: parameters dictionary
        """
        return cirq.resolve_parameters(self.give_circuit(idx)[0], res)


class Evaluator(Basic):
    def __init__(self, args, info=None, loading=False, nrun_load=0):
        """

        This class serves as evaluating the energy, admiting the new circuit or not. Also stores the results either if there's a relevant modification or not. Finally, it allows for the possibilty of loading previous results, an example for the TFIM is:

            %load_ext autoreload
            %autoreload 2
            from utilities.circuit_basics import Evaluator
            evaluator = Evaluator(loading=True, args={"n_qubits":3, "J":4.5})
            unitary, energy, indices, resolver = evaluator.raw_history[47]


        """
        if not loading:
            super(Evaluator, self).__init__(n_qubits=args.n_qubits)
            self.raw_history = {}
            self.evolution = {}
            self.lowest_energy = None
            self.directory = self.create_folder(args,info)
            self.displaying = "\n hola, soy VANS :), and it's {} \n".format(datetime.now())

        else:
            super(Evaluator, self).__init__(n_qubits=args["n_qubits"])
            args_load={}
            for str,default in zip(["n_qubits", "J", "g","noise","problem"], [3,0.,1.,0.,"TFIM"]):
                if str not in list(args.keys()):
                    args_load[str] = default
                else:
                    args_load[str] = args[str]
            self.load(args_load,nrun=nrun_load)


    def create_folder(self,args, info):
        if not os.path.exists(args.problem):
            os.makedirs(args.problem)
        if float(args.noise) > 0:
            if not os.path.exists(args.problem+"/noisy"):
                os.makedirs(args.problem+"/noisy")
            name_folder = args.problem+"/noisy/"+str(args.n_qubits)+"Q - J "+str(args.J)+" g "+str(args.g)+ " noise "+str(args.noise)
        else:
            name_folder = args.problem+"/"+str(args.n_qubits)+"Q - J "+str(args.J)+" g "+str(args.g)
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
            nrun=0
            final_folder = name_folder+"/run_"+str(nrun)
            with open(name_folder+"/runs.txt", "w+") as f:
                f.write(info)
                f.close()
            os.makedirs(final_folder)
        else:
            folder = os.walk(name_folder)
            nrun=0
            for k in list(folder)[0][1]:
                if k[0]!=".":
                    nrun+=1
            final_folder = name_folder+"/run_"+str(nrun)
            with open(name_folder+"/runs.txt", "r") as f:
                a = f.readlines()[0]
                f.close()
            with open(name_folder+"/runs.txt", "w") as f:
                f.write(str(nrun)+"\n")
                f.write(info)
                f.write("\n")
                f.close()
            os.makedirs(final_folder)
        return final_folder

    def load(self,args, nrun=0):
        if float(args["noise"]) > 0:
            name_folder = args["problem"]+"/noisy/"+str(args["n_qubits"])+"Q - J "+str(args["J"])+" g "+str(args["g"])+ " noise "+str(args["noise"])
        else:
            name_folder = args["problem"]+"/"+str(args["n_qubits"])+"Q - J "+str(args["J"])+" g "+str(args["g"])
        self.load_dicts_and_displaying(name_folder+"/run_"+str(nrun))
        return

    def save_dicts_and_displaying(self):
        output = open(self.directory+"/raw_history.pkl", "wb")
        pickle.dump(self.raw_history, output)
        output.close()
        output = open(self.directory+"/evolution.pkl", "wb")
        pickle.dump(self.evolution, output)
        output.close()
        with open(self.directory+"/evolution.txt","w") as f:
            f.write(self.displaying)
            f.close()
        return

    def load_dicts_and_displaying(self,folder):

        with open(folder+"/raw_history.pkl" ,"rb") as h:
            self.raw_history = pickle.load(h)
        with open(folder+"/evolution.pkl", "rb") as hh:
            self.evolution = pickle.load(hh)
        with open(folder+"/evolution.txt", "r") as f:
            a = f.readlines()
            f.close()
        self.displaying = a
        return self.displaying

    def accept_energy(self, E, noise=False):
        """
        in the presence of noise, don't give gates for free!

        E: energy after some optimization (to be accepted or not)
        """
        if self.lowest_energy is None:
            return True
        else:
            # return E < self.lowest_energy
            return (E-self.lowest_energy)/np.abs(self.lowest_energy) < 0.01

        # if noise:
            # return E < self.lowest_energy
        # else:
            # return (E-self.lowest_energy)/np.abs(self.lowest_energy) < 0.01

    def add_step(self,indices, resolver, energy, relevant=True):
        """
        indices: list of integers describing circuit to save
        resolver: dictionary with the corresponding circuit's symbols
        energy: expected value of target hamiltonian on prepared circuit.
        relevant: if energy was minimized on that step
        """
        if self.lowest_energy is None:
            self.lowest_energy = energy
        elif energy < self.lowest_energy:
            self.lowest_energy = energy

        self.raw_history[len(list(self.raw_history.keys()))] = [self.give_unitary(indices, resolver), energy, indices, resolver, self.lowest_energy]
        if relevant == True:
            self.evolution[len(list(self.evolution.keys()))] = [self.give_unitary(indices, resolver), energy, indices,resolver, self.lowest_energy]
        return



class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            print("hey")
            np.seed(datetime.now().microsecond + datetime.now().second)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator
