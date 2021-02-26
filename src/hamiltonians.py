import pennylane as qml

# Create a (ferromagnetic) transverse-field Ising Hamiltonian.
def tfi_chain(num_qubits, h, periodic=False,g=0):
    obs    = [qml.PauliZ(q) @ qml.PauliZ(q+1) for q in range(num_qubits-1)]
    coeffs = [-1.0]*(num_qubits-1)
    
    if periodic:
        obs.append(qml.PauliZ(0) @ qml.PauliZ(num_qubits-1))
        coeffs.append(-1.0)


    if abs(h) > 1e-15:
        obs.extend([qml.PauliX(q) for q in range(num_qubits)])
        coeffs.extend([-h]*num_qubits)

    if abs(g) > 1e-15:
        obs.extend([qml.PauliZ(q) for q in range(num_qubits)])
        coeffs.extend([-g]*num_qubits)

    return qml.Hamiltonian(coeffs, obs)
