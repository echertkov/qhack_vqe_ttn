import pennylane as qml
from pennylane import numpy as np
import numpy as np2
import matplotlib.pyplot as plt
from ttn import ttn_circuit, get_num_ttn_gates
from hea import hea_circuit
from mera import mera_circuit, get_num_mera_gates
from hamiltonians import tfi_chain

import pandas as pd

seed = 2

num_params_per_gate = 15

num_samples = 50
qubits      = [16]

hs          = [0.0,0.5,0.8,0.9,1.0,1.1,1.2,1.5,2.0]
g = 0.5 # Longitudinal field

initializations = ["Random |0...0>", "Random |+...+>", "Zero |0...0>",  "Zero |+...+>"]

data_dicts = []

for num_qubits in qubits:
    print(f"=== Number of qubits = {num_qubits} ===")
    dev = qml.device("default.qubit.autograd", wires=num_qubits)

    constant_hea_depth = 3
    
    ansatze           = [ttn_circuit(num_qubits),
                         mera_circuit(num_qubits),
                         hea_circuit(num_qubits),
                         hea_circuit(num_qubits),
                         hea_circuit(num_qubits)
                         ]
    num_params_ansatz = [num_params_per_gate * get_num_ttn_gates(num_qubits), 
                         num_params_per_gate * get_num_mera_gates(num_qubits),
                         3 * (num_qubits * constant_hea_depth),
                         3 * (num_qubits * int(np.floor(np.log2(num_qubits)))),
                         3 * (num_qubits * num_qubits)]
    ansatz_names      = ["TTN", 
                        "MERA", 
                        "HEA (constant)", 
                        "HEA (log)", 
                        "HEA (linear)"]

    for (ansatz, ansatz_name, num_params) in zip(ansatze, ansatz_names, num_params_ansatz):
        print(f" --- {ansatz_name} ({num_params} parameters) ---")

        for init in initializations:
            print(f"  +++ {init} +++")
            mod_ansatz = ansatz
            if init in ["Random |+...+>", "Zero |+...+>"]:
                @qml.template
                def ansatz_plus(params, wires):
                    for q in range(num_qubits):
                        qml.Hadamard(q)
                    ansatz(params, wires)

                mod_ansatz = ansatz_plus
            
            for h in hs:
                print(f"   ||| h={h} |||")

                H = tfi_chain(num_qubits, h, g=g)
                grad_vals = []

                cost_fn = qml.ExpvalCost(mod_ansatz, H, dev, optimize=True)
                grad    = qml.grad(cost_fn)

                # Use the same random numbers for each set of parameters/ansatz.
                np.random.seed(seed)
                for i in range(num_samples):
                    if init in ["Zero |0...0>", "Zero |+...+>"]:
                        params = np.zeros(num_params)
                    else:
                        params = np.pi*(np.random.rand(num_params) - 1.0)

                    if "HEA" in ansatz_name:
                        params = np.reshape(params, (len(params)//3, 3))
                    
                    # Convert the info to normal numpy arrays, not the special pennlyane.numpy tensors.
                    
                    gradient  = np2.array(grad(params)[0]).flatten()
                    params    = np2.array(params).flatten()

                    data_dict = {
                        "num_qubits" : num_qubits,
                        "ansatz_name": ansatz_name,
                        "h"          : h,
                        "init"       : init,
                        "params"     : params,
                        "grad"       : gradient,
                        "sample"     : i
                    }

                    #print(data_dict)

                    data_dicts.append(data_dict)

                    if init in ["Zero |0...0>", "Zero |+...+>"]:
                        break

            # Save and overwrite the Pandas dataframe for every initialization.
            df = pd.DataFrame(data_dicts)
            df.to_pickle("tfi_longitudinal_gradients_df_3.p")





