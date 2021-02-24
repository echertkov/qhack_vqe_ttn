import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N",          help="# qubits",type=int)
parser.add_argument("--h",          help="h",type=float)
parser.add_argument("--ansatz",     help="Which ansatz (ttn,mera,hea)")
parser.add_argument("--opt",        help="Which otimizer (qng,adagrad,adam)")
parser.add_argument("--step",       help="Optimizer step size(0.05)",type=float,default=0.05)
parser.add_argument("--initState",  help="Which initial state (0/+)", default="0")
parser.add_argument("--randomize",  help="Randomize initial params?", type=int,default=1)
parser.add_argument("--two_qubit",  help="Which two qubit gate (two_qubit,arbitrary,simple6,simple2)")
parser.add_argument("--periodic",   help="Is the mera periodic?",type=int,default=False)
parser.add_argument("--fix_layers", help="Is the mera have fixed_layers?",type=int,default=True)
args = parser.parse_args()
import time
import numpy as np
import pennylane as qml
from ttn   import ttn_circuit
from mera  import mera_circuit, get_num_mera_gates, arb_qubit_gate
from hea   import hea_circuit
from gates import two_qubit_gate,simple_two_qubit_gate1,simple_two_qubit_gate2
from hamiltonians import tfi_chain
from observables  import sigma_z,sigma_x


num_qubits = int(args.N)

# The simulator.
dev = qml.device('default.qubit', wires=num_qubits, analytic=True)



h = float(args.h)
H         = tfi_chain(num_qubits, h, periodic=False)
sz        = sigma_z(num_qubits)
sx        = sigma_x(num_qubits)
print(H)
print("----")
print(sz)
print("----")
print(sx)

# The TFI model at the critical point.

if args.two_qubit == "two_qubit":
    unitary_parameterization = two_qubit_gate
    num_params_per_gate = 15
elif args.two_qubit == "simple6":
    unitary_parameterization = simple_two_qubit_gate1
    num_params_per_gate = 6
elif args.two_qubit == "simple2":
    unitary_parameterization = simple_two_qubit_gate2
    num_params_per_gate = 2
elif args.two_qubit == "arbitrary":
    unitary_parameterization = arb_qubit_gate
    num_params_per_gate = 15
else:
    raise NotImplementedError("Two Qubit Parameterization not found!")


# Choose which ansatz
if args.ansatz == "ttn":
    depth               = int(np.floor(np.log2(num_qubits))) # The depth of the TTN.
    num_gates           = num_qubits - 1                     # The number of two-qubit gates in the TTN.
    num_params          = num_params_per_gate * num_gates    # The total number of parameters in the TTN.
    
    base_ansatz         = ttn_circuit(num_qubits, unitary_parameterization)
elif args.ansatz == "mera":
    periodic            = args.periodic == 1
    fix_layers          = args.fix_layers == 1

    depth               = int(np.floor(np.log2(num_qubits))) # The depth of the TTN.
    num_gates           = get_num_mera_gates(num_qubits, periodic, fix_layers)
    num_params          = num_params_per_gate * num_gates
    
    base_ansatz     = mera_circuit(num_qubits,unitary_parameterization,periodic=periodic,fix_layers=fix_layers)
elif args.ansatz == "hea":
    hea_depth   = 3
    num_params  = (num_qubits * hea_depth - 1)*3
    base_ansatz = hea_circuit(num_qubits,two_qubit_gate)
else:
    raise NotImplementedError("Not implemented yet")

print(f"Total # of Parameters={num_params}")

# determine initial state
initState = args.initState
if initState == "0":
    ansatz = base_ansatz
elif initState == "+":
    @qml.template
    def ansatz_f(params, wires):
        for q in range(num_qubits):
            qml.Hadamard(q)
        base_ansatz(params,wires)
    ansatz = ansatz_f
else:
    raise NotImplementedError("Not found initial state!")

step_size = args.step

params = np.loadtxt(f"params_{args.ansatz}_{args.two_qubit}_{num_qubits}_{h}_{args.opt}_{step_size}_{initState}{args.randomize}.txt")
print(f"{args.ansatz}_{args.two_qubit}_{num_qubits}_{h}_{args.opt}_{step_size}_{initState}{args.randomize}")
print(params)

if args.ansatz == 'hea' and params.shape[1]!=3: params = np.reshape(params, ((params)//3, 3))

# The circuit for computing the expectation value of H.
ee_fn = qml.ExpvalCost(ansatz, H, dev)
sz_fn = qml.ExpvalCost(ansatz, sz, dev)
sx_fn = qml.ExpvalCost(ansatz, sx, dev)
ee_obs = ee_fn(params)
sz_obs = sz_fn(params)
sx_obs = sx_fn(params)

print(ee_obs / num_qubits)
print(sz_obs)
print(sx_obs)


np.savetxt(f"obs_{args.ansatz}_{args.two_qubit}_{num_qubits}_{h}_{args.opt}_{step_size}_{initState}{args.randomize}.txt",[ee_obs/num_qubits,sz_obs,sx_obs])



