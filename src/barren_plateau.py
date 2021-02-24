import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mera import mera_circuit, get_num_mera_gates
from hamiltonians import tfi_chain

num_samples = 200
qubits = [2, 4, 8]
variances = []

np.random.seed(1)
h = 1.0
periodic=False
fix_layers=True


for num_qubits in qubits:
    H = tfi_chain(num_qubits, h)
    grad_vals = []
    for i in range(num_samples):
        print(f"num_qubits {num_qubits}, step {i}")
        dev = qml.device("default.qubit", wires=num_qubits)
        ansatz = mera_circuit(num_qubits, periodic, fix_layers)
        cost_fn = qml.ExpvalCost(ansatz, H, dev)
        grad = qml.grad(cost_fn)


        num_params_per_gate = 15
        num_gates           = get_num_mera_gates(num_qubits, periodic, fix_layers)
        num_params          = num_params_per_gate * num_gates
        params = np.pi*(np.random.rand(num_params) - 1.0)

        gradient = np.array(grad(params)[0])
        grad_vals.append(gradient)
    variances.append(np.mean(np.var(grad_vals, axis=0)))

print(variances)
#variances = np.array(np.mean(variances, axis=1))
qubits = np.array(qubits)

# Fit the semilog plot to a straight line
p = np.polyfit(qubits, np.log(variances), 1)


# Plot the straight line fit to the semilog
plt.semilogy(qubits, variances, "o")
plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.", label="Slope {:3.2f}".format(p[0]))
plt.xlabel(r"N Qubits")
plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
plt.legend()
plt.show()



