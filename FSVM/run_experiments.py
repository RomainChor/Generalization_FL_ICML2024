import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.generalization.models import generalization_rounds, compute_bound
from utils.generalization.dataloaders import load_binary_mnist

warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size':20})
sns.set_style('whitegrid')
palette = sns.color_palette("tab10")
sns.set_palette(palette)




def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./")
    parser.add_argument("--save_path", type=str, default="save/")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "plot"],
                        help="whether to run simulations ('train') or to plot figures ('plot').")
    parser.add_argument("--comparison", type=str, default="K", choices=["K", "n"],
                        help="to run comparison for different 'K' or 'n'.")
    parser.add_argument("--MC", type=int, default=10,
                        help="number of runs (Monte-Carlo simulations)")
    
    parser.add_argument("--classes", nargs="+", type=int, default=[1,6],
                        help="MNIST classes")
    parser.add_argument("--frac_iid", type=float, default=0,
                        help="fraction of clients with AWGN")
    parser.add_argument("--iid_std", type=float, default=0.2,
                        help="standard deviation of the AWGN")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = args_parser()

    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path
    
    data_params = {
        "name":"mnist", 
        "proj_dim":4000, 
        "gamma":0.05,
        "seed":args.seed
    }

    params = {
        "task":"classification",
        "loss":"hinge", 
        "frac_iid":args.frac_iid, 
        "iid_std":args.iid_std,
        "lr":0.01,
        "seed":args.seed
    }

    mnist = load_binary_mnist(class1=args.classes[0], class2=args.classes[1], params=data_params, 
                              path=args.data_path) # Preprocess and load two classes of MNIST

    compare = args.comparison

    if params["frac_iid"] > 0:
        SAVE_PATH += "/non_iid/"
    if compare == "K":
        n = 100 
        K_values = [10, 20, 50]
    else:
        K = 10
        n_values = [100, 200, 500]
    R_values = [1, 2, 3, 4, 5, 10, 15]

    if args.mode == "train": # Run simulations and save results
        df_list = []
        if compare == "K":
            e_values = [40, 40, 38]
            for K, e in zip(K_values, e_values):
                data_params["N"] = n*K
                params["client_epochs"] = e
                df = generalization_rounds(K, R_values, data_params, params, mnist, args.MC)
                df.to_pickle(SAVE_PATH+"df_K{}.pickle".format(K))
                df_list.append(df)
        else:
            e_values = [50, 50, 50]
            for n, e in zip(n_values, e_values):
                data_params["N"] = n*K
                params["client_epochs"] = e
                df = generalization_rounds(K, R_values, data_params, params, mnist, args.MC)
                df.to_pickle(SAVE_PATH+"df_n{}.pickle".format(n))
                df_list.append(df)

    # =========================================================================
    # =========================================================================
    elif args.mode == "plot": # Import saved values to create plots
        df_list = []
        q_values = [0.7]*len(R_values)

        values = K_values if compare == "K" else n_values
        file = "df_"+compare

        fig1, ax1 = plt.subplots(1, 2, figsize=(18, 6))
        fig2, ax2 = plt.subplots(1, 2, figsize=(18, 6))
        plot_params = {"marker":'o', "ms":9, "mew":2, "lw":4}
        for i, x in enumerate(values):
            df = pd.read_pickle(SAVE_PATH+file+"{}.pickle".format(x))
            if compare == "K":
                u = n
                v = x
            else:
                u = x
                v = K
            df["bound"] = np.array([compute_bound(u, v, R, q, theta=0.2) 
                                    for R, q in zip(R_values, q_values)])

            ax1[0].plot(
                R_values, df["fed_gen"],      
                **plot_params, c=palette[i],
                label=compare+r"$ = {}$".format(x)
            )
            ax1[1].plot(
                R_values, df["bound"],
                **plot_params, ls='--', alpha=0.8, c=palette[i],
                label=compare+r"$ = {}$".format(x)
            )

            ax2[0].plot(
                R_values, df["fed_emp_risks_01"],     
                **plot_params, c=palette[i],
                label=compare+r"$ = {}$".format(x)
            )        

            ax2[1].plot(
                R_values, df["fed_risks_01"],     
                **plot_params, c=palette[i],
                label=compare+r"$ = {}$".format(x)
            )

        xticks = np.arange(0, np.max(R_values)+2, 2, dtype=int)
        ax1[0].set(xlabel=r"$R$", ylabel=r"$\operatorname{gen}(s, \overline{w})$", 
                   xticks=xticks)
        ax1[0].tick_params(which='minor', length=5)
        ax1[0].grid(ls='--', alpha=0.8)
        ax1[0].legend()

        ax1[1].set(xlabel=r"$R$", ylabel=r"Bound",
                   xticks=xticks)
        ax1[1].set_ylim(bottom=0)
        ax1[1].tick_params(which='minor', length=5)
        ax1[1].grid(ls='--', alpha=0.8)
        ax1[1].legend()

        ax2[0].set(xlabel=r"$R$", ylabel=r"$\mathcal{\hat L}(s, \overline{w})$",
                   xticks=xticks)
        ax2[0].tick_params(which='minor', length=5)
        ax2[0].grid(ls='--', alpha=0.8)
        ax2[0].legend()

        ax2[1].set(xlabel=r"$R$", ylabel=r"$\mathcal{L}(\overline{w})$",
                   xticks=xticks)
        ax2[1].set_yticks(ax2[0].get_yticks())
        ax2[1].tick_params(which='minor', length=5)
        ax2[1].grid(ls='--', alpha=0.8)
        ax2[1].legend()

        # Save plots
        z = n if compare == "K" else K
        file = "n" if compare == "K" else "K"
        fig1.tight_layout()
        fig1.savefig(SAVE_PATH+"gen_flsvm_"+file+"{}.png".format(z))

        fig2.tight_layout()
        fig2.savefig(SAVE_PATH+"risk_flsvm_"+file+"{}.png".format(z))

    else:
        n = 100
        K = 20
        data_params = {
            "name":"mnist", 
            "N":n*K, # Can be larger than "real" datasets' size
            "proj_dim":4000, # Dimension of the projection space for SVM
            "gamma":0.05, # Gaussian kernel parameter
            "seed":args.seed
        }

        params = {
            "task":"classification",
            "loss":"hinge", # Loss function
            "lr":0.01, # Learning rate (initial)
            "client_epochs":40, # Client's model number of epochs
            "n_rounds":10, #...
            "frac_iid":0,
            "seed":args.seed
        }

        R_values = [1, 2, 3, 4, 5, 10, 15, 20, 50, 80, 100]
        for K in [20, 50]:
            data_params = n*K
            q_values = []
            for R in R_values:
                params["n_rounds"] = R
                q_r = compute_q(K, data_params, params, mnist, M)
                q_values.append(np.mean(q_r))

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(R_values, q_values, lw=3)
            ax.set(xlabel=r"$R$", ylabel=r"$q_{e,b}$", xticks=np.arange(0, 110, 10));
            ax.set_xscale('log')
            ax.set(xticks=[1, 10, 100], xticklabels=[1, 10, 100])
            ax.grid(which='both', ls='--', alpha=0.8)
            fig.savefig(SAVE_PATH+"coeff_estim_K{}.png".format(K))