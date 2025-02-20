import json
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from src.data import InputsData, SignalData
from src.model import LinearTimeInvariantModel
from src.optimizer import BayesianEDA

# Define the path to the data directory
data_dir = "../data/Insomnia_SCR_MATLAB"

# Get the list of subjects
folders = os.listdir(data_dir)
subject_list = [folder for folder in folders if not folder.startswith(".")]
subject_list.sort()

# Define the list of phases
phase_list = ["cond", "ext", "recall"]

# Get the list of excluded data
exclusion_path = "../data/data_exclusion.json"
with open(exclusion_path, "r") as f:
    exclusion_file = json.load(f)
exclusion_list = exclusion_file["missing"] + exclusion_file["noisy"] + exclusion_file["over_stimuli"]

# Main loop
for subject, phase in product(subject_list, phase_list):
    # Skip excluded data
    if subject in exclusion_list:
        continue

    # Load data
    data_path = os.path.join(data_dir, subject, f"{subject}_{phase}.mat")
    data_file = loadmat(data_path)["data"]
    y_obs = data_file[:, 0]
    u_obs = data_file[:, 1]

    # Process data
    y_obs = SignalData(
        raw_data=y_obs,
        original_frequency=200,
        downsampled_frequency=4,
        lowpass_cutoff_frequency=2,
        lowpass_filter_order=4096,
        outlier_window_duration=3,
    )
    u_obs = InputsData(raw_data=u_obs, original_frequency=200, downsampled_frequency=4)

    #! Debug
    y_obs.data = y_obs.data * 0.1

    # Define the model
    eta = 0.2
    #! Debug
    # A = lambda tau: np.array(
    #     [
    #         [-1 / tau[0], 0, 0],
    #         [(1 - eta) / tau[0], -1 / tau[1], 0],
    #         [eta / tau[0], 0, -1 / tau[2]],
    #     ]
    # )
    # B = np.array([1, 0, 0]).reshape(3, 1)
    # C = np.array([0, 1, 1]).reshape(1, 3)
    A = lambda tau: np.array(
        [
            [-1 / tau[0], 0, 0],
            [1 / tau[0], -1 / tau[1], 0],
            [1 / tau[0], 0, -1 / tau[2]],
        ]
    )
    B = np.array([1, 0, 0]).reshape(3, 1)
    C = np.array([0, 1 - eta, eta]).reshape(1, 3)
    model = LinearTimeInvariantModel(A=A, B=B, C=C)

    # Set constraints where C theta <= b
    constraint_C = np.array(
        [
            [-1, 0, 0],  # rise time > 0.2
            [0, -1, 0],  # fast decay time > 0.1
            [0, 0, -1],  # slow decay time > 6
            [2, -1, 0],  # fast decay time > 2 * rise time
            [0, 2, -1],  # slow decay time > 2 * fast decay time
        ]
    )
    constraint_b = np.array(
        [
            -0.2,  # rise time > 0.2
            -0.1,  # fast decay time > 0.1
            -6,  # slow decay time > 6
            0,  # fast decay time > 2 * rise time
            0,  # slow decay time > 2 * fast decay time
        ]
    )

    # Set priors
    tau_mean = np.array(
        [
            6.5,  # rise time mean
            70,  # fast decay time mean
            200,  # slow decay time means
        ]
    )
    tau_stdev = np.array(
        [
            2,  # rise time stdev
            5,  # fast decay time stdev
            8,  # slow decay time stdev
        ]
    )

    # Initialize parameters
    # tau = [tau_r, tau_p, tau_d]
    tau_0 = np.zeros(3)
    tau_0[0] = 6
    tau_0[1] = tau_0[0] * 8
    tau_0[2] = tau_0[1] * 15

    # Initial state
    x0 = np.array([0, 0, y_obs[0]])

    # Estimate parameters, states and inputs
    optimizer = BayesianEDA(
        model=model,
        constraints_C=constraint_C,
        constraints_b=constraint_b,
        theta_mean=tau_mean,
        theta_variance=tau_stdev**2,
    )
    theta_log, x_log, u_log = optimizer.fit(theta0=tau_0, x0=x0, y=y_obs)

    # Plot results
    x = x_log[-1]  # (t, 3)
    u = u_log[-1]  # (t, 1)

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(y_obs.t, y_obs.data, label="observed")
    ax[0].plot(y_obs.t, np.einsum("ij,tj->t", C, x[1:]), label="predicted")
    ax[1].plot(y_obs.t, x[1:, 2], label="tonic")
    ax[2].plot(y_obs.t, x[1:, 1], label="phasic")
    ax[3].stem(y_obs.t, u[:, 0], label="CS")
    plt.legend()
    plt.show()
