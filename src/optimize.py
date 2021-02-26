import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N",          help="# qubits",type=int)
parser.add_argument("--h",          help="h",type=float)
parser.add_argument("--ansatz",     help="Which ansatz (ttn,mera,hea,healog,healin)")
parser.add_argument("--opt",        help="Which otimizer (qng,adagrad,adam)")
parser.add_argument("--step",       help="Optimizer step size(0.05)",type=float,default=0.05)
parser.add_argument("--initState",  help="Which initial state (0/+)", default="0")
parser.add_argument("--randomize",  help="Randomize initial params?(2=continue)", type=int,default=1)
parser.add_argument("--two_qubit",  help="Which two qubit gate (two_qubit,arbitrary,simple6,simple2)")
parser.add_argument("--periodic",   help="Is the mera periodic?",type=int,default=False)
parser.add_argument("--fix_layers", help="Is the mera have fixed_layers?",type=int,default=True)
parser.add_argument("--rtol",       help="rtol",type=float,default=1e-5)
parser.add_argument("--atol",       help="atol",type=float,default=1e-9)
parser.add_argument("--maxiter",    help="maximum iterations",type=int,default=100)
args = parser.parse_args()

import time
import numpy as np
import pennylane as qml
from ttn   import ttn_circuit
from mera  import mera_circuit, get_num_mera_gates, arb_qubit_gate
from hea   import hea_circuit
from gates import two_qubit_gate,simple_two_qubit_gate1,simple_two_qubit_gate2
from hamiltonians import tfi_chain


num_qubits = int(args.N)

# The simulator.
dev = qml.device('default.qubit', wires=num_qubits)

# The TFI model at the critical point.
h = float(args.h)
H = tfi_chain(num_qubits, h, periodic=args.periodic)
print(H)

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
elif args.ansatz == "healog":
    hea_depth   = 3
    num_params  = 3 * (num_qubits * int(np.floor(np.log2(num_qubits))))
    base_ansatz = hea_circuit(num_qubits,two_qubit_gate)
elif args.ansatz == "healin":
    hea_depth   = 3
    num_params  = 3 * (num_qubits * num_qubits)
    base_ansatz = hea_circuit(num_qubits,two_qubit_gate)
else:
    raise NotImplementedError("Not implemented yet")

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

# The circuit for computing the expectation value of H.
cost_fn = qml.ExpvalCost(ansatz, H, dev, optimize=True)

print(f"Total # of Parameters={num_params}")

# Perform VQE.
step_size = args.step
if args.opt == "qng":
    opt = qml.QNGOptimizer(stepsize=step_size, diag_approx=True, lam=0.1)
elif args.opt=="adagrad":
    opt = qml.AdagradOptimizer(stepsize=step_size) 
elif args.opt=="adam":
    opt = qml.AdamOptimizer(stepsize=step_size)
else:
  raise NotImplementedError("Optimizer not found")
# Initialize the parameters.
np.random.seed(1)
if args.randomize==1:
    params = np.pi*(np.random.rand(num_params) - 1.0)
elif args.randomize==2:
    params = np.loadtxt(f"params_{args.ansatz}_{args.two_qubit}_{num_qubits}_{h}_{args.opt}_{step_size}_{initState}{1}.txt")
    print(len(params),num_params)
    assert len(params)==num_params
else:
    params = np.zeros([num_params])

if args.ansatz[:3] == 'hea': params = np.reshape(params, (len(params)//3, 3))


# Optimizer parameters.
rtol    = args.rtol
atol    = args.atol
maxiter = args.maxiter

print_results = True

Es       = []
prev_obj = 0.0
start = time.perf_counter()
for step in range(maxiter):
    params, obj  = opt.step_and_cost(cost_fn, params)
    diff_obj     = np.abs(prev_obj - obj)
    rel_obj      = diff_obj/np.abs(prev_obj + 1e-15)
    Es.append(obj/num_qubits)
    prev_obj = obj

    if print_results:
        print(f"Step {step+1}, <H>/L = {obj/num_qubits}")

    if rel_obj < rtol or diff_obj < atol:
        if print_results:
            print(f"Took {step+1} steps.")
            print(f"diff_obj = {diff_obj}, rel_obj = {rel_obj}.")
        break
    
stop = time.perf_counter()
if print_results:
    print(f'Final params = {params}')

if h==1:
    # According to this website: https://dmrg101-tutorial.readthedocs.io/en/latest/tfim.html 
    # (though their Hamiltonian has a typo: it should be Pauli matrices not spin operators.)
    exact_E0 = 1.0 - 1.0/np.sin(np.pi/(2.0 * (2.0 * num_qubits + 1.0))) 
    if print_results:
        print(f'Exact <H>/L = {exact_E0/num_qubits}')

print(f"total time={stop-start}")
np.savetxt(f"params_{args.ansatz}_{args.two_qubit}_{num_qubits}_{h}_{args.opt}_{step_size}_{initState}{args.randomize}.txt",params)
np.savetxt(f"energies_{args.ansatz}_{args.two_qubit}_{num_qubits}_{h}_{args.opt}_{step_size}_{initState}{args.randomize}.txt",Es)
