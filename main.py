import os

import numpy as np
import pandas as pd
from plot import plot_results
from scipy.io import loadmat, savemat
from src.data import InputsData, SignalData
from src.model import LinearTimeInvariantModel
from src.optimizer import BayesianEDA

# Define the path to save results
exp_name = "best"
save_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"
save_path_fig = f"results/plots/{exp_name}/{{subject}}_{{phase}}.pdf"

# Define the path to load data
load_path = "../data/Insomnia_SCR_MATLAB/{subject}/{subject}_{phase}.mat"

# Get the list of subjects and phases
datalist_path = "../data/datalist.csv"
with open(datalist_path, "r") as f:
    datalist = pd.read_csv(f)
datalist = datalist[datalist["status"].isin(["included"])]  # Exclude data by visual inspection
datalist = datalist[datalist["group"].isin(["ID", "GS"])]  # Exclude data that is not ID or GS

# Main loop
for _, row in datalist.iterrows():
    subject = row["subject"]
    phase = row["phase"]

    print(f"Processing {subject} {phase}...")

    # Load data
    _load_path = load_path.format(subject=subject, phase=phase)
    data_file = loadmat(_load_path)["data"]
    y_obs = data_file[:, 0]
    u_obs = data_file[:, 1]

    # Process data
    y_obs = SignalData(raw_data=y_obs, original_frequency=200, outlier_window_duration=3)
    u_obs = InputsData(raw_data=u_obs, original_frequency=200)

    # Define the model
    eta = 0.3  # Ratio of tonic to phasic
    A = lambda tau: np.array(
        [
            [-1 / tau[0], 0, 0],  # duct
            [(1 - eta) / tau[0], -1 / tau[1], 0],  # phasic
            [eta / tau[0], 0, -1 / tau[2]],  # tonic
        ]
    )
    B = np.array([1, 0, 0]).reshape(3, 1)
    C = np.array([0, 1, 1]).reshape(1, 3)
    model = LinearTimeInvariantModel(A=A, B=B, C=C)

    # Set constraints 0.1 <= tau_r <= tau_f/1.5 <= tau_s/1.5^2
    constraint_A = np.array([[1, 0, 0], [-1.5, 1, 0], [0, -1.5, 1]])
    constraint_lb = np.array([0.1, 0, 0])

    # Set priors [tau_r, tau_f, tau_s]
    tau_mean = np.array([1, 25, 120])
    tau_stdev = np.array([1, 1, 2]) * 1e-5

    # Initialize parameters with the mean of the prior
    tau_0 = tau_mean

    # Initial state [duct, phasic, tonic]
    x0 = np.array([0, 0, y_obs[0].item()])

    # Estimate parameters, states and inputs
    optimizer = BayesianEDA(
        model=model,
        constraint_A=constraint_A,
        constraint_lb=constraint_lb,
        theta_mean=tau_mean,
        theta_variance=tau_stdev**2,
        input_threshold=0.01 * np.min(y_obs),
        max_iter=100,
        tolerance_theta=0.01,
        tolerance_input=0.01,
    )
    theta_log, x_log, u_log = optimizer(theta0=tau_0, x0=x0, y=y_obs)

    # Save results
    results = {
        "tau_r": theta_log[-1][0],
        "tau_f": theta_log[-1][1],
        "tau_s": theta_log[-1][2],
        "t": y_obs.t,
        "y_obs": y_obs.data[:, 0],
        "phasic": x_log[-1][:, 1],
        "tonic": x_log[-1][:, 2],
        "u": u_log[-1][:, 0],
        "u_obs": u_obs.data[:, 0],
    }
    _save_path = save_path.format(subject=subject, phase=phase)
    os.makedirs(os.path.dirname(_save_path), exist_ok=True)
    savemat(_save_path, results)

    _fig_save_path = save_path_fig.format(subject=subject, phase=phase)
    os.makedirs(os.path.dirname(_fig_save_path), exist_ok=True)
    plot_results(
        t=results["t"],
        y_obs=results["y_obs"],
        phasic=results["phasic"],
        tonic=results["tonic"],
        u=results["u"],
        tau_r=results["tau_r"],
        tau_f=results["tau_f"],
        tau_s=results["tau_s"],
        u_obs=None,  # Don't plot u_obs
        title=f"{subject} {phase}",
        save_path=_fig_save_path,
    )
