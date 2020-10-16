import numpy as np
from utilities.variational import VQE
import cirq
g=0.75
js=np.linspace(-1.1, 1.1,20)
j=.4

v = VQE(n_qubits=4, g=g,J=j, problem="xxz")
print(cirq.unitary(cirq.Circuit(v.observable)))
