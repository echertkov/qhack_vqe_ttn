# Hooked on Photonics
## Tackling quantum phase transitions and barren plateaus in VQE with tensor networks

```   
    ____  _    _          _____ _  __           ___   ___ ___  __ 
  / __ \| |  | |   /\   / ____| |/ /          |__ \ / _ \__ \/_ |
 | |  | | |__| |  /  \ | |    | ' /   ______     ) | | | | ) || |
 | |  | |  __  | / /\ \| |    |  <   |______|   / /| | | |/ / | |
 | |__| | |  | |/ ____ \ |____| . \            / /_| |_| / /_ | |
  \___\_\_|  |_/_/    \_\_____|_|\_\          |____|\___/____||_|
````
---

### Project Description:


One of the most promising applications of variational quantum algorithms is the study of condensed matter phenomena, such as quantum phase transitions, with near-term quantum computers. Yet a major challenge in the successful application of variational quantum algorithms is the barren plateau phenomenon, where gradients become exponentially small in the number of qubits. While barren plateaus have been observed for certain types of variational quantum circuits and cost functions, it is unclear whether the phenomenon would significantly hinder the simulation of condensed matter systems. Our goal in this project is to explore this problem and develop strategies for avoiding barren plateaus in the study of quantum phase transitions.

In particular, we use the variational quantum eigensolver (VQE) to find the ground states of the transverse field Ising model, a spin chain whose ground state is known to undergo a quantum phase transition. To avoid the barren plateau phenomenon in our analysis of this model, we train variational circuits with physically relevant structure, such as tree tensor networks (TTN) and the multi-scale entanglement renormalization ansatz (MERA). The MERA, a tensor network used to study quantum critical systems, is particularly well-suited to our task. We show that for our problem TTN’s and MERA’s generally produce larger gradients than a hardware-efficient ansatz (HEA) typically used in VQE and thereby are easier to train and help alleviate barren plateaus.


### Preliminary Data

At 8 qubits, we successfully train a MERA tensor network, such that we can even use observables to spot the phase transition   

<img src="https://github.com/echertkov/qhack_vqe_ttn/raw/main/images/observables.png" width="550px" />  

We also notice that when searching for barren plateaus, the TTN and MERA do not suffer, as compared with a HEA with linear depth   

<img src="https://github.com/echertkov/qhack_vqe_ttn/raw/main/images/avgvar_vs_N_h_1.0.png" width="550px" />   
