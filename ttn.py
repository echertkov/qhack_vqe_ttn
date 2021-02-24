import pennylane as qml
import numpy as np

# Creates a function representing the tree-tensor
# network circuit with the given number of qubits and
# the given type of two-qubit gate.
def ttn_circuit(num_qubits, two_qubit_gate, fix_layers=False):
    # The depth of the TTN.
    # The connectivity of the two qubit gates in the TTN.

    depth = int(np.ceil(np.log2(num_qubits)))

    free_qubits = set(list(range(num_qubits)))
    pattern  = []
    distance = 2**(depth - 1)
    for layer in range(depth):
        for q1 in range(0, num_qubits, 2*distance):
            q2 = q1 + distance
            if q2 < num_qubits:
                pattern.append([q1, q2])

                if q1 in free_qubits:
                    free_qubits.remove(q1)
                if q2 in free_qubits:
                    free_qubits.remove(q2)

        distance = distance // 2

    for q in free_qubits:
        pattern.append([q-1,q])

    @qml.template
    def ttn(params, wires):
        # Reformat the parameters list into a list of lists.
        # Each two-qubit gate should receive a list of parameters.
        parameters = []
        if fix_layers:
            num_params_per_layer = len(params) // depth
            for layer in range(depth):
                num_gates_in_layer = 2**layer
                params_gate        = [params[j] for j in range(layer*num_params_per_layer, (layer+1)*num_params_per_layer)]
                parameters.extend([params_gate] * num_gates_in_layer)
        else:
            num_params_per_gate = len(params) // len(pattern)
            parameters          = [[params[i*num_params_per_gate+j] for j in range(num_params_per_gate)] for i in range(len(pattern))]
        
        # Apply the two qubit gates into the pattern of a tree.
        qml.broadcast(two_qubit_gate, wires, pattern, parameters=parameters, kwargs=None)

    return ttn
