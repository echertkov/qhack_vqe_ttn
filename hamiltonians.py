import pennylane as qml

# Create a (ferromagnetic) transverse-field Ising Hamiltonian.
def tfi_chain(num_qubits, h, periodic=False):
    obs = [qml.PauliZ(q) @ qml.PauliZ(q+1) for q in range(num_qubits-1)]
    if periodic:
        obs.append(qml.PauliZ(0) @ qml.PauliZ(num_qubits-1))
    obs.extend([qml.PauliX(q) for q in range(num_qubits)])

    if periodic:
        coeffs = [-1.0]*num_qubits + [-h]*num_qubits
    else:
        coeffs = [-1.0]*(num_qubits-1) + [-h]*num_qubits

    return qml.Hamiltonian(coeffs, obs)
