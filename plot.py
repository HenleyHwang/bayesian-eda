import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat


def plot_results(_save_path, t, y_obs, phasic, tonic, u, tau_r, tau_f, tau_s):
    fig, ax = plt.subplots(4, 1, figsize=(6, 6))
    fig.suptitle(f"$\\tau_r = {tau_r:.2f}$, $\\tau_f = {tau_f:.2f}$, $\\tau_s = {tau_s:.2f}$")
    ax[0].plot(t, y_obs, label="observed")
    ax[0].plot(t, phasic + tonic, label="predicted")
    ax[1].plot(t, phasic, label="phasic")
    ax[2].plot(t, tonic, label="tonic")
    ax[3].stem(t, u, label="activations", basefmt=" ", markerfmt=" ")
    ax[1].legend()
    ax[2].legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(_save_path), exist_ok=True)
    fig.savefig(_save_path)
    plt.close(fig)


if __name__ == "__main__":
    exp_name = "best"
    load_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"
    save_path = f"results/plots/{exp_name}/{{subject}}_{{phase}}.png"

    # Get the list of subjects and phases
    datalist_path = "../data/datalist.csv"
    with open(datalist_path, "r") as f:
        datalist = pd.read_csv(f)
    datalist = datalist[datalist["status"].isin(["included"])]
    datalist = datalist[datalist["group"].isin(["ID", "GS"])]

    # Main loop
    for row in datalist.itertuples():
        subject = row.subject
        phase = row.phase

        # Load result
        _load_path = load_path.format(subject=subject, phase=phase)
        result = loadmat(_load_path)
        t = result["t"].flatten()
        y_obs = result["y_obs"].flatten()
        u_obs = result["u_obs"].flatten()
        phasic = result["phasic"].flatten()
        tonic = result["tonic"].flatten()
        u = result["u"].flatten()
        tau_r = result["tau_r"].item()
        tau_f = result["tau_f"].item()
        tau_s = result["tau_s"].item()

        # Plot
        _save_path = save_path.format(subject=subject, phase=phase)
        plot_results(_save_path, t, y_obs, phasic, tonic, u, tau_r, tau_f, tau_s)
