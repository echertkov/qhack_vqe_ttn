import pennylane as qml
import numpy as np
from pennylane.templates.subroutines import ArbitraryUnitary

def get_num_mera_gates(num_qubits, periodic=False, fix_layers=False):
    # The depth of the MERA.
    depth = int(np.ceil(np.log2(num_qubits)))
    if fix_layers:
        return 2*depth - 1

    # The connectivity of the two qubit gates in the MERA.
    N = 1
    distance = 2**(depth - 1)

    #add top
    distance = distance//2

    for layer in range(1,depth):

        # add isometry
        for q1 in range(0, num_qubits, 2*distance):
            q2 = q1 + distance
            if q2 < num_qubits:
                N += 1
        #add unitary
        for q2 in range(distance, num_qubits, 2*distance):
            q3 = q2 + distance
            if periodic or q3 < num_qubits:
                N += 1
        distance = distance // 2
    return N

@qml.template
def arb_qubit_gate(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14, wires):
    ArbitraryUnitary([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14],wires)


# Creates a function representing the MERA
# network circuit with the given number of qubits and
# the given type of two-qubit gate.
def mera_circuit(num_qubits,two_qubit_gate=arb_qubit_gate, periodic=False, fix_layers=False):
    # The depth of the MERA.
    depth = int(np.ceil(np.log2(num_qubits)))

    # The connectivity of the two qubit gates in the MERA.
    pattern  = []
    distance = 2**(depth - 1)

    #add top
    pattern.append([0,distance])
    distance = distance//2

    for layer in range(1,depth):

        # add isometry
        for q1 in range(0, num_qubits, 2*distance):
            q2 = q1 + distance
            if q2 < num_qubits:
                pattern.append([q1, q2])

        #add unitary
        for q2 in range(distance, num_qubits, 2*distance):
            q3 = q2 + distance
            if periodic or q3 < num_qubits:
                pattern.append([q2,q3 % num_qubits])
        distance = distance // 2

    @qml.template
    def mera(params, wires):
        # Reformat the parameters list into a list of lists.
        # Each two-qubit gate should receive a list of parameters.
        # Apply the two qubit gates into the pattern of a tree.

        num_gates = len(pattern)

        if fix_layers:
            gate_size = len(params)//(2*depth-1)  #determine 2qubit gate size
            parameters = []
            parameters.append(params[:gate_size])
            for layer in range(1,depth):
                gates_per_layer = 2**layer
                #isometry
                for i in range(gates_per_layer):
                    parameters.append(params[gate_size*(2*layer-1):gate_size*2*layer])
                #unitary
                for i in range(gates_per_layer - int(not periodic)):
                    parameters.append(params[gate_size*2*layer:gate_size*(2*layer+1)])
        else:
            parameters = params
            gate_size = len(params)//num_gates # determine 2qubit gate size

        parameters = np.reshape(parameters, [num_gates, gate_size])
        qml.broadcast(unitary=two_qubit_gate, pattern=pattern, wires=wires, parameters=parameters,\
                kwargs=None)

    return mera

'''
 if __name__ == "__main__":
     depth = 3
     num_qubits = 2**depth
     periodic=True
     fix_layers = True
     dev = qml.device('default.qubit', wires=num_qubits)
     mera_template = mera_circuit(num_qubits, periodic=periodic, fix_layers=fix_layers)
     num_gates = get_num_mera_gates(num_qubits, periodic=periodic, fix_layers=fix_layers)
     num_params = num_gates*15
     np.random.seed(1)
     params = np.pi*(np.random.rand(num_params) - 1.0)
     @qml.qnode(dev)
     def circuit(params):
         mera_template(params, num_qubits)
         return qml.expval(qml.PauliZ(0))
     circuit(params)
     print(circuit.draw())
'''
