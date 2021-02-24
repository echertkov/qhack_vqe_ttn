import pennylane as qml

# compute average of sigma_z
def sigma_z(num_qubits):
    obs = [qml.PauliZ(q) for q in range(num_qubits)]
    coeffs = [-1.0/num_qubits]*(num_qubits)

    return qml.Hamiltonian(coeffs, obs)
# compute average of sigma_z

def sigma_x(num_qubits):
    obs = [qml.PauliX(q) for q in range(num_qubits)]
    coeffs = [-1.0/num_qubits]*(num_qubits)

    return qml.Hamiltonian(coeffs, obs)
