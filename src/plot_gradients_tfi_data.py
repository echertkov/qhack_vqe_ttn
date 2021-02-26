import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd

# The main dataset for the transverse field Ising model.
df1       = pd.read_pickle("tfi_gradients_df_N2_to_N16.p")
df2       = pd.read_pickle("tfi_gradients_df_N18_all.p")
df        = pd.concat([df1,df2], ignore_index=True)
run_name  = "_N4_to_N18"

# An extra dataset for the TFI model plus a g = 0.5 longitudinal field.
#df1 = pd.read_pickle("tfi_longitudinal_gradients_df_1.p")
#df2 = pd.read_pickle("tfi_longitudinal_gradients_df_2.p")
#df3 = pd.read_pickle("tfi_longitudinal_gradients_df_3.p")
#df  = pd.concat([df1,df2,df3], ignore_index=True)
#run_name = "_longitudinal_123"

Ns = [4,6,8,10,12,14,16,18] #np.sort(df['num_qubits'].unique())

initializations = ["Random |0...0>", "Random |+...+>" , "Zero |0...0>",  "Zero |+...+>"]
line_styles     = ['-', '--', ':', '-.']

ansatz_names    = ["TTN", "MERA", "HEA (constant)", "HEA (log)", "HEA (linear)"]

hs = np.sort(df['h'].unique())

ind_param = -1

"""
num_qubits = 16

# Plot distribution of gradients for random inits.
for ansatz_name in ansatz_names:
    plt.figure()
    plt.title(f"{ansatz_name}")
    for h in hs:
        mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == "Random |0...0>") & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == num_qubits)

        grads = df[mask]['grad']

        grads0 = [grads.iloc[s][0] for s in range(grads.size)]

        plt.hist(grads0, bins=20, density=True, label=f"$h = {h}$", alpha=0.9)
    plt.xlabel("$\\partial_0 \\langle H\\rangle$")
    plt.ylabel("Frequency")
    plt.grid()
    plt.legend()
"""

"""
# Plot means of gradients for a single parameter vs h.
plt.figure()
for ansatz_name in ansatz_names:
    for (init, ls) in zip(initializations, line_styles):
        if "Zero" in init:
            continue

        vars_grad0 = []
        for h in hs:
            mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == num_qubits)

            grads = df[mask]['grad']

            grads0     = [grads.iloc[s][ind_param] for s in range(grads.size)]
            var_grad0  = np.mean(grads0)
            vars_grad0.append(var_grad0)

        plt.plot(hs, vars_grad0, ls=ls, label=f"{ansatz_name}, {init}")
        plt.ylabel("$Avg[\\partial_0 \\langle H\\rangle]$")
        plt.xlabel("$h$")
        plt.grid()
        plt.legend()

# Plot variances of gradients for a single parameter vs h.
plt.figure()
for ansatz_name in ansatz_names:
    for (init, ls) in zip(initializations, line_styles):
        if "Zero" in init:
            continue

        vars_grad0 = []
        for h in hs:
            mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == num_qubits)

            grads = df[mask]['grad']

            grads0     = [grads.iloc[s][ind_param] for s in range(grads.size)]
            var_grad0  = np.var(grads0)
            vars_grad0.append(var_grad0)

        plt.plot(hs, vars_grad0, ls=ls, label=f"{ansatz_name}, {init}")
        plt.ylabel("$Var[\\partial_0 \\langle H\\rangle]$")
        plt.xlabel("$h$")
        plt.grid()
        plt.legend()

# Plot variances of gradients for a single parameter vs N
for h in hs:
    plt.figure()
    plt.title(f'$h = {h}$')
    for ansatz_name in ansatz_names:
        for (init, ls) in zip(initializations, line_styles):
            if "Zero" in init:
                continue

            vars_grad0 = []
            for N in Ns:
                mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == N)

                grads = df[mask]['grad']

                grads0     = [grads.iloc[s][ind_param] for s in range(grads.size)]
                var_grad0  = np.var(grads0)
                vars_grad0.append(var_grad0)

            plt.plot(Ns, vars_grad0, 'o-', ls=ls, label=f"{ansatz_name}, {init}")
            #plt.xscale('log')
            plt.yscale('log')

            plt.ylabel("$Var[\\partial_0 \\langle H\\rangle]$")
            plt.xlabel("$N$")
            plt.grid()
            plt.legend()


# Plot variances of gradients for all the individual parameters vs N
for h in hs:
    plt.figure()
    plt.title(f'$h = {h}$')
    for ansatz_name in ansatz_names:
        for (init, ls) in zip(initializations, line_styles):
            if "Zero" in init:
                continue

            for ind_p in range(6):
                vars_grad0 = []
                for N in Ns:
                    mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == N)

                    grads = df[mask]['grad']

                    grads0     = [grads.iloc[s][ind_p] for s in range(grads.size)]
                    var_grad0  = np.var(grads0)
                    vars_grad0.append(var_grad0)

                plt.plot(Ns, vars_grad0, 'o-', ls=ls, label=f"{ansatz_name}, {init}, {ind_p}")
            #plt.xscale('log')
            plt.yscale('log')

            plt.ylabel("$Var[\\partial_0 \\langle H\\rangle]$")
            plt.xlabel("$N$")
            plt.grid()
            plt.legend(fontsize=6)

"""

# Plot scalings of avg variances of gradients vs h
plt.figure()
for ansatz_name in ansatz_names:
    if ansatz_name not in ["TTN", "MERA", "HEA (log)"]:
        continue
    for (init, ls) in zip(initializations, line_styles):
        if "Zero" in init or "|+...+>" in init:
            continue

        plot_hs = []
        alphas  = []
        for h in hs:
            plot_Ns        = []
            mean_var_grads = []
            for N in Ns:
                mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == N)

                grads = df[mask]['grad']
                if grads.size == 0:
                    continue

                num_params  = grads.iloc[0].size

                var_grad_params = np.zeros(num_params)
                for ind_p in range(num_params):
                    grad_params            = [grads.iloc[s][ind_p] for s in range(grads.size)]
                    var_grad_params[ind_p] = np.var(grad_params)

                plot_Ns.append(N)
                mean_var_grads.append(np.mean(var_grad_params))

            if len(plot_Ns) == 0:
                continue

            p = np.polyfit(np.log(plot_Ns), np.log(mean_var_grads), 1)

            slope = p[0]
            alpha = -slope
            alphas.append(alpha)
            plot_hs.append(h)

        plt.plot(plot_hs, alphas, ls=ls, label=f"{ansatz_name}")

plt.title("$Avg[Var[\\partial \\langle H\\rangle]] = B N^{-\\nu}$ fit")
plt.axvline(1.0, color='k', ls="--")
plt.ylabel("Exponent $\\nu$")
plt.xlabel("Transverse field $h$")
plt.grid()
plt.legend()
plt.savefig(f"plots/alphas_vs_h{run_name}.png", dpi=500)

# Plot the average variance of the gradient wrt to a single parameter vs h.
plt.figure()
num_qubits = 18
for ansatz_name in ansatz_names:
    for (init, ls) in zip(initializations, line_styles):
        if "Zero" in init or "|+...+>" in init:
            continue

        plt.title(f"$N={num_qubits}$ qubits") #Longitudinal field $g=0.5$ 
        mean_var_grads = []
        plot_hs        = []
        for h in hs:
            mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == num_qubits)

            grads       = df[mask]['grad']
            num_samples = grads.size
            if num_samples == 0:
                continue

            num_params  = grads.iloc[0].size

            var_grad_params = np.zeros(num_params)
            for ind_p in range(num_params):
                grad_params            = [grads.iloc[s][ind_p] for s in range(grads.size)]
                var_grad_params[ind_p] = np.var(grad_params)

            plot_hs.append(h)
            mean_var_grads.append(np.mean(var_grad_params))

        plt.plot(plot_hs, mean_var_grads, ls=ls, label=f"{ansatz_name}")
        plt.ylabel("Avg. variance of gradient $Avg[Var[\\partial \\langle H\\rangle]]$")
        plt.xlabel("Transverse field $h$")
        plt.grid()
        plt.legend()
plt.savefig(f"plots/avgvar_vs_h{run_name}.png", dpi=500)

# Plot average-variance vs N
for h in hs:
    plt.figure()
    plt.title(f'Transverse field $h = {h}$') #, longitudinal field $g=0.5$')
    for ansatz_name in ansatz_names:
        for (init, ls) in zip(initializations, line_styles):
            if "Zero" in init or "|+...+>" in init:
                continue
        
            mean_var_grads = []
            plot_Ns        = []
            for N in Ns:
                mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == N)

                grads       = df[mask]['grad']
                num_samples = grads.size
                if num_samples == 0:
                    continue

                num_params  = grads.iloc[0].size

                var_grad_params = np.zeros(num_params)
                for ind_p in range(num_params):
                    grad_params            = [grads.iloc[s][ind_p] for s in range(grads.size)]
                    var_grad_params[ind_p] = np.var(grad_params)

                plot_Ns.append(N)
                mean_var_grads.append(np.mean(var_grad_params))

            label = f"{ansatz_name}"

            # Perform a polynomial fit for the TTN and MERA.
            if ansatz_name in ["TTN", "MERA", "HEA (log)"]:
                p     = np.polyfit(np.log(plot_Ns), np.log(mean_var_grads), 1)
                poly  = np.poly1d(p)
                x_fit = np.linspace(np.min(Ns), np.max(Ns), 100)
                y_fit = np.exp(poly(np.log(x_fit)))
                plt.plot(x_fit, y_fit, 'k--')

                B  = np.exp(p[1])
                nu = -p[0]
                label += f", $Var = ({B:0.4f}) N^{{-{nu:0.4f} }}$"

            # Perform an exponential fit for the linear depth HEA
            if ansatz_name in ["HEA (linear)"]:
                p     = np.polyfit(plot_Ns, np.log(mean_var_grads), 1)
                poly  = np.poly1d(p)
                x_fit = np.linspace(np.min(Ns), np.max(Ns), 100)
                y_fit = np.exp(poly(x_fit))
                plt.plot(x_fit, y_fit, 'r--')

                B   = np.exp(p[1])
                chi = -1.0/p[0]
                label += f", $Var = ({B:0.4f}) exp(-N/({chi:0.4f}))$"

            plt.plot(plot_Ns, mean_var_grads, 'o-', ls=ls, label=label) # , {init}

    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xticks(ticks=Ns, labels=[f"{N}" for N in Ns])
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    
    plt.ylabel("Avg. variance of gradient $Avg[Var[\\partial \\langle H\\rangle]]$")
    plt.xlabel("Number of qubits $N$")
    plt.grid()
    plt.legend()
    plt.savefig(f"plots/avgvar_vs_N_h_{h}{run_name}.png", dpi=500)

# Plot the average abs of the average of the gradient wrt to a single parameter vs h.
plt.figure()
plt.title(f"$N={num_qubits}$ qubits") #Longitudinal field $g=0.5$ 
for ansatz_name in ansatz_names:
    for (init, ls) in zip(initializations, line_styles):
        if "Zero" in init or "|+...+>" in init:
            continue

        plot_hs             = []
        mean_abs_mean_grads = []
        for h in hs:
            mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == num_qubits)

            grads       = df[mask]['grad']
            num_samples = grads.size
            if num_samples == 0:
                continue
            num_params  = grads.iloc[0].size

            mean_grad_params = np.zeros(num_params)
            for ind_p in range(num_params):
                grad_params            = [grads.iloc[s][ind_p] for s in range(grads.size)]
                mean_grad_params[ind_p] = np.mean(grad_params)

            plot_hs.append(h)
            mean_abs_mean_grads.append(np.mean(np.abs(mean_grad_params)))

        plt.plot(plot_hs, mean_abs_mean_grads, ls=ls, label=f"{ansatz_name}") # , {init}
        plt.ylabel("Avg. magnitude of gradient $Avg[|Avg[\\partial \\langle H\\rangle]|]$")
        plt.xlabel("Transverse field $h$")
        plt.grid()
        plt.legend()
plt.savefig(f"plots/avgabsavg_vs_h{run_name}.png", dpi=500)

"""
# Plot variances of gradient norms vs h.
plt.figure()
for ansatz_name in ansatz_names:
    for (init, ls) in zip(["Random |0...0>", "Random |+...+>"], line_styles):
        vars_grad0 = []
        for h in hs:
            mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == num_qubits)

            grads = df[mask]['grad']
            grads0     = [norm(grads.iloc[s]) for s in range(grads.size)]

            var_grad0  = np.var(grads0)
            vars_grad0.append(var_grad0)

        plt.plot(hs, vars_grad0, ls=ls, label=f"{ansatz_name}, {init}")
        plt.ylabel("$Var[||\\partial \\langle H\\rangle||]$")
        plt.xlabel("$h$")
        plt.grid()
        plt.legend()

# Plot norms of average gradients vs h.
plt.figure()
for ansatz_name in ansatz_names:
    for (init, ls) in zip(initializations, line_styles):
        vars_grad0 = []
        for h in hs:
            mask = (df['ansatz_name'] == ansatz_name) & (df['init'] == init) & (np.abs(df['h'] - h) < 1e-15) & (df['num_qubits'] == num_qubits)

            avg_grad = df[mask]['grad'].mean(axis=0)

            var_grad0 = norm(avg_grad)
            vars_grad0.append(var_grad0)

        plt.plot(hs, vars_grad0, ls=ls, label=f"{ansatz_name}, {init}")
        plt.ylabel("$||Avg[\\partial \\langle H\\rangle]||$")
        plt.xlabel("$h$")
        plt.grid()
        plt.legend()
"""

plt.show()