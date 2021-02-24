import pennylane as qml

# An arbitrary one-qubit gate. (3 parameters)
def one_qubit_gate(p0, p1, p2, q):
    qml.RX(p0, wires=[q])
    qml.RY(p1, wires=[q])
    qml.RZ(p2, wires=[q])

# An arbitrary two-qubit gate. (15 parameters)
@qml.template
def two_qubit_gate(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14, wires):
    # This is a parametrization of an arbitrary two-qubit gate
    # that is called the Cartan decomposition.
    one_qubit_gate(p0,p1,p2, wires[0])
    one_qubit_gate(p3,p4,p5, wires[1])

    # Rxx
    qml.Hadamard(wires=[wires[0]])
    qml.Hadamard(wires=[wires[1]])
    qml.MultiRZ(p6, wires=wires)
    qml.Hadamard(wires=[wires[0]])
    qml.Hadamard(wires=[wires[1]])

    # Ryy
    qml.inv(qml.S(wires=[wires[0]]))
    qml.Hadamard(wires=[wires[0]])
    qml.inv(qml.S(wires=[wires[1]]))
    qml.Hadamard(wires=[wires[1]])
    qml.MultiRZ(p7, wires=wires)
    qml.Hadamard(wires=[wires[0]])
    qml.S(wires=[wires[0]])
    qml.Hadamard(wires=[wires[1]])
    qml.S(wires=[wires[1]])
    
    # Rzz
    qml.MultiRZ(p8, wires=wires)

    one_qubit_gate(p9,p10,p11, wires[0])
    one_qubit_gate(p12,p13,p14, wires[1])

# A simple parametrized two-qubit gate with six parameters.
@qml.template
def simple_two_qubit_gate1(p0,p1,p2,p3,p4,p5, wires):
    one_qubit_gate(p0,p1,p2, wires[0])
    one_qubit_gate(p3,p4,p5, wires[1])
    qml.CNOT(wires=wires)

# Another simple parametrized two-qubit gate with two parameters.
# (This one is designed to produce a purely real wave function.)
@qml.template
def simple_two_qubit_gate2(p0,p1, wires):
    qml.RY(p0, wires=wires[0])
    qml.RY(p1, wires=wires[1])
    qml.CNOT(wires=wires)
