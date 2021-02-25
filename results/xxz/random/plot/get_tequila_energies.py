import tequila as tq
import numpy as np
from tqdm import tqdm


def HXXZ(num_qubits,delta,lam):
    # PBC
    ham = tq.paulis.X(num_qubits-1)*tq.paulis.X(0)
    ham += tq.paulis.Y(num_qubits-1)*tq.paulis.Y(0)
    ham += delta*tq.paulis.Z(num_qubits-1)*tq.paulis.Z(0)
    ham += lam*tq.paulis.Z(num_qubits-1)
    for i in range(num_qubits-1):
        ham += tq.paulis.X(i)*tq.paulis.X(i+1)
        ham += tq.paulis.Y(i)*tq.paulis.Y(i+1)
        ham += delta*tq.paulis.Z(i)*tq.paulis.Z(i+1)
        ham += lam*tq.paulis.Z(i)
    return(ham)3

def exact(num_qubits,delta,lam):
    ham_matrix = HXXZ(num_qubits,delta,lam).to_matrix()
    energ = np.linalg.eigvals(ham_matrix)
    return(min(energ))

energs = []
for ani in tqdm(np.arange(0,3.1,.1)):
    energs.append(exact(8,ani,1.0))
np.real(energs)
