import numpy as np
import cirq

def dict_to_json(dictionary):
    d="{"
    for k,v in dictionary.items():
        if isinstance(k,str):
            d+='\"{}\":\"{}\",'.format(k,v)
        else:
            d+='\"{}\":{},'.format(k,v)
    d=d[:-1]
    d+="}" #kill the comma
    return "\'"+d+ "\'"

def compute_ground_energy(obse,qubits):
    ind_to_2 = {"0":np.eye(2), "1":cirq.unitary(cirq.X), "2":cirq.unitary(cirq.Y), "3":cirq.unitary(cirq.Z)}
    ham = np.zeros((2**len(qubits),2**len(qubits))).astype(np.complex128)
    for kham in obse:
        item= kham.dense(qubits)
        string = item.pauli_mask
        matrices = [ind_to_2[str(int(ok))] for ok in string]
        ham += give_kr_prod(matrices)*item.coefficient
    return np.sort(np.real(np.linalg.eigvals(ham)))


def give_kr_prod(matrices):
    #matrices list of 2 (or more in principle) matrices
    while len(matrices) != 1:
        sm, smf=[],[]
        for ind in range(len(matrices)):
            sm.append(matrices[ind])
            if ind%2==1 and ind>0:
                smf.append(np.kron(*sm))
                sm=[]
        matrices = smf
    return matrices[0]
