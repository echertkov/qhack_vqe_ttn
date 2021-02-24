import pennylane as qml
import numpy as np

# Creates a function representing the hareware efficient ansatz
# network circuit with the given number of qubits and
# the given type of two-qubit gate.
def hea_circuit(num_qubits, two_qubit_gate):

    @qml.template
    def hea(params, wires):

        n_qubits = len(wires)
        n_rotations = len(params)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            #n_extra_rots = n_rotations - n_layers * n_qubits

            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            # There may be "extra" parameter sets required for which it's not necessarily
            # to perform another full alternating cycle. Apply these to the qubits as needed.
            #extra_params = params[-n_extra_rots:, :]
            #extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
            #qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])

    return hea
