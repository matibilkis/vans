def cirq_friendly_observable(obs):
    PAULI_BASIS = {
        'I': np.eye(2),
        'X': np.array([[0., 1.], [1., 0.]]),
        'Y': np.array([[0., -1j], [1j, 0.]]),
        'Z': np.diag([1., -1]),
    }

    pauli3 = cirq.linalg.operator_spaces.kron_bases(PAULI_BASIS, repeat=3)
    decomp = cirq.linalg.operator_spaces.expand_matrix_in_orthogonal_basis(obs, pauli3)

    PAULI_BASIS_CIRQ = {
        'I': cirq.X,
        'X': cirq.X,
        'Y': cirq.Y,
        'Z': cirq.Z,
    }

    unt = []
    for term in decomp.items():
        gate_name = term[0]
        coeff = term[1]
        s = 0
        ot = float(coeff)
        for qpos, single_gate in enumerate(gate_name):
            if single_gate == "I":
                ot *= PAULI_BASIS_CIRQ[single_gate](qubits[qpos])*PAULI_BASIS_CIRQ[single_gate](qubits[qpos])
            else:
                ot *= PAULI_BASIS_CIRQ[single_gate](qubits[qpos])
        if s < 3:
            unt.append(ot)
    return unt
