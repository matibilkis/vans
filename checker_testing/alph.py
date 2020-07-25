self.indexed_cnots = {}
count = 0
for control in range(self.n_qubits):
    for target in range(self.n_qubits):
        if control != target:
            self.indexed_cnots[str(count)] = [control, target]
            count += 1
self.number_of_cnots = len(self.indexed_cnots)
#int(np.math.factorial(self.n_qubits)/np.math.factorial(self.n_qubits -2))

# Create one_hot alphabet
self.alphabet_gates = [cirq.CNOT, cirq.ry, cirq.rx(-np.pi/2), cirq.I]
self.alphabet = []

alphabet_length = self.number_of_cnots + (len(self.alphabet_gates)-1)*self.n_qubits
for ind, k in enumerate(range(self.number_of_cnots + (len(self.alphabet_gates)-1)*self.n_qubits)): #5 accounts for 2 CNOTS and 3 other ops
    one_hot_gate = [-1]*alphabet_length
    one_hot_gate[ind] = 1
    self.alphabet.append(one_hot_gate)
