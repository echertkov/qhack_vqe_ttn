import numpy as np

import pennylane as qml

from ttn import ttn_circuit
from hea import hea_circuit
from gates import two_qubit_gate
from hamiltonians import tfi_chain

num_qubits = 8

# The simulator.
dev = qml.device('default.qubit', wires=num_qubits)

# The TFI model at the critical point.
h = 1.0
H = tfi_chain(num_qubits, h)
print(H)

# The hareware efficient ansatz circuit.
#ansatz = ttn_circuit(num_qubits, two_qubit_gate)
ansatz = hea_circuit(num_qubits, two_qubit_gate)

# The circuit for computing the expectation value of H.
cost_fn = qml.ExpvalCost(ansatz, H, dev)


# Initialize the parameters.
np.random.seed(1)
#num_param_sets = (2 ** num_qubits) - 1
hea_depth = 2
num_param_sets = hea_depth * num_qubits 
params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))



# Perform VQE.
opt = qml.AdagradOptimizer(stepsize=0.2) # The optimizer to use.

# Optimizer parameters.
rtol    = 1e-5
atol    = 1e-9
maxiter = 100

print_results = True

prev_obj = 0.0
for step in range(maxiter):
    params, obj  = opt.step_and_cost(cost_fn, params)
    diff_obj     = np.abs(prev_obj - obj)
    rel_obj      = diff_obj/np.abs(prev_obj + 1e-15)

    prev_obj = obj

    if print_results:
        print(f"Step {step+1}, <H>/L = {obj/num_qubits}")

    if rel_obj < rtol or diff_obj < atol:
        if print_results:
            print(f"Took {step+1} steps.")
            print(f"diff_obj = {diff_obj}, rel_obj = {rel_obj}.")
        break
    
if print_results:
    print(f'Final params = {params}')

# According to this website: https://dmrg101-tutorial.readthedocs.io/en/latest/tfim.html 
# (though their Hamiltonian has a typo: it should be Pauli matrices not spin operators.)
exact_E0 = 1.0 - 1.0/np.sin(np.pi/(2.0 * (2.0 * num_qubits + 1.0))) 
if print_results:
    print(f'Exact <H>/L = {exact_E0/num_qubits}')
