import openfermionpyscf as ofpyscf
import cirq
import numpy as np
import openfermion as of
import json


class OpenFermion_to_Cirq:
    """
    takes an of.QuitOperator and maps it to list of cirq.PauliStrings ready to be fed into tfq model
    """

    def __init__(self):
        pass

    def str_to_gate(self,string):
        if string == "":
            return cirq.I
        elif string == "Z":
            return cirq.Z
        elif string == "X":
            return cirq.X
        elif string == "Y":
            return cirq.Y
        else:
            print("problemo!")
            return cirq.I

    def __call__(self, qham, qubits):
        cir=[]
        for ops, val in qham.terms.items():
            if len(ops) == 0:
                cir.append(float(val)*cirq.I.on(qubits[0]))
            else:
                gate = ops[0]
                g=float(val)*self.str_to_gate(gate[-1]).on(qubits[gate[0]])
                for gate in ops[1:]:
                    g*=self.str_to_gate(gate[-1]).on(qubits[gate[0]])
                cir.append(g)
        return cir

    def get_matrix_rep(self, observable, qubits):
        """
        method that gives the matrix representation (as a sanity check)
        """
        ind_to_2 = {"0":np.eye(2), "1":cirq.unitary(cirq.X), "2":cirq.unitary(cirq.Y), "3":cirq.unitary(cirq.Z)}
        mat = np.zeros((2**len(qubits), 2**len(qubits))).astype(np.complex128)
        for oper in observable:
            item = oper.dense(qubits)
            string = item.pauli_mask
            matrix = np.kron(*[np.kron(*[ind_to_2[str(int(ok))] for ok in string[:2]]), np.kron(*[ind_to_2[str(int(ok))] for ok in string[2:]])])
            mat+=item.coefficient*matrix
        return mat


class ChemicalObservable(OpenFermion_to_Cirq):

    def __init__(self):
        """
        geometry: molecular geometry, example for H2 molecule:  [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
        qubits: cirq.GridQubits or similar
        """
        super(ChemicalObservable).__init__()

    def load_geometry_correct_format(self, geometry):
        """
        changes string of list of tuples to list, preserving things that OpenFermion likes (yes, I've been an entire afternoon fighting with this :E)
        """

        newgeometry = []
        skip_next=False
        for ind, k in enumerate(list(geometry)):
            if k.isalpha() and not skip_next:
                newgeometry.append('\'')
                newgeometry.append(k)
                if not list(geometry)[ind+1].isalpha():
                    newgeometry.append('\'')
                else:
                    newgeometry.append(geometry[ind+1])
                    newgeometry.append('\'')
                    skip_next = True
            else:
                newgeometry.append(k)

        newgeometry = "".join(newgeometry)
        newgeometry= eval(newgeometry)
        return newgeometry

    def give_observable(self,qubits, geometry, multiplicity=1, charge=0, basis="sto-3g"):
        """
        To-Do: implement a bool if the number of cirq qubits is not enough for the jordan_wigner
        """

        geometry = self.load_geometry_correct_format(geometry)

        hamiltonian = ofpyscf.generate_molecular_hamiltonian(geometry=geometry, basis=str(basis),
                                                             multiplicity=int(multiplicity), charge=int(charge))
        qham = of.jordan_wigner(hamiltonian)
        return self(qham, qubits)
