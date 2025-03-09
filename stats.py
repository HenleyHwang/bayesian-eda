import os
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ranksums, wilcoxon

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["cm"]
plt.rcParams["font.size"] = 12


if __name__ == "__main__":
    exp_name = "best"
    load_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"

    save_path = f"results/paper/stats"
    os.makedirs(save_path, exist_ok=True)

    # Dictionary for display
    units = {
        "tau_r": "s",
        "tau_f": "s",
        "tau_s": "s",
        "u_num": "1",
        "u_bar": r"$\mu$S/s",
    }
    symbols = {
        "tau_r": r"$\tau_r$",
        "tau_f": r"$\tau_f$",
        "tau_s": r"$\tau_s$",
        "u_num": r"$\#\mathbf{u}_{>0}$",
        "u_bar": r"$\bar{\mathbf{u}}_{>0}$",
    }
    phase_names = {
        "cond": "Conditioning",
        "ext": "Extinction",
        "recall": "Recall",
    }

    # Get the list of subjects and phases
    datalist_path = "../data/datalist.csv"
    with open(datalist_path, "r") as f:
        datalist = pd.read_csv(f)
    datalist = datalist[datalist["group"].isin(["ID", "GS"])]

    # Exclude over-stimulated subjects
    def over_stimulated(row):
        if row["status"] != "included":
            return
        result = loadmat(load_path.format(subject=row["subject"], phase=row["phase"]))
        if np.sum(result["u_obs"] > 0) > 50:
            row["status"] = "excluded"

    datalist.apply(over_stimulated, axis=1)

    # Get the list of phases
    phases = datalist["phase"].unique()

    # Get number of subjects for each phase and group
    rows = []
    for phase in phases:
        n_ID = ((datalist["status"] == "included") & (datalist["phase"] == phase) & (datalist["group"] == "ID")).sum()
        n_GS = ((datalist["status"] == "included") & (datalist["phase"] == phase) & (datalist["group"] == "GS")).sum()
        rows.append({"phase": phase, "ID": n_ID, "GS": n_GS})
    df_n_subjects = pd.DataFrame(rows, columns=["phase", "ID", "GS"])
    df_n_subjects.to_csv(os.path.join(save_path, "n_subjects.csv"), index=False)

    # Initialize dictionary {group: {phase: {feature: [value]}}}
    features = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Extract features
    for _, row in datalist.iterrows():
        subject = row["subject"]
        phase = row["phase"]
        group = row["group"]
        status = row["status"]

        if status == "included":
            # Load result
            _load_path = load_path.format(subject=subject, phase=phase)
            result = loadmat(_load_path)

            tau_r = result["tau_r"].item()
            tau_f = result["tau_f"].item()
            tau_s = result["tau_s"].item()

            u = result["u"].flatten()
            u_pos = u[u > 0]
            u_num = len(u_pos)
            u_bar = np.mean(u_pos) if u_num > 0 else 0

        else:
            # Use nan for excluded data for pairwise tests
            tau_r, tau_f, tau_s, u_num, u_bar = np.nan, np.nan, np.nan, np.nan, np.nan

        # Extract features
        features[group][phase]["tau_r"].append(tau_r)
        features[group][phase]["tau_f"].append(tau_f)
        features[group][phase]["tau_s"].append(tau_s)
        features[group][phase]["u_num"].append(u_num)
        features[group][phase]["u_bar"].append(u_bar)

    # Convert to arrays
    for group in features.keys():
        for phase in features[group].keys():
            for feature in features[group][phase].keys():
                features[group][phase][feature] = np.array(features[group][phase][feature])

    # Box plots
    for phase in phases:
        for feature in features["ID"][phase].keys():
            ID_feature = features["ID"][phase][feature]
            GS_feature = features["GS"][phase][feature]
            ID_feature = ID_feature[~np.isnan(ID_feature)]
            GS_feature = GS_feature[~np.isnan(GS_feature)]
            plt.figure(figsize=(3, 3))
            plt.boxplot([ID_feature, GS_feature], tick_labels=["ID", "GS"])
            plt.ylabel(f"{symbols[feature]} ({units[feature]})")
            plt.title(f"Box Plot of {symbols[feature]} in {phase_names[phase]} Phase")
            plt.savefig(os.path.join(save_path, f"{feature}_{phase}.pdf"), bbox_inches="tight")
            plt.close()

    # Features for each phase
    for phase in phases:
        rows = []
        for feature in features["ID"][phase].keys():
            ID_feature = features["ID"][phase][feature]
            GS_feature = features["GS"][phase][feature]
            ID_feature = ID_feature[~np.isnan(ID_feature)]
            GS_feature = GS_feature[~np.isnan(GS_feature)]
            rows.append(
                {
                    "feature": f"{symbols[feature]} ({units[feature]})",
                    "IDmedian": np.median(ID_feature),
                    "IDlower": np.percentile(ID_feature, 25),
                    "IDupper": np.percentile(ID_feature, 75),
                    "GSmedian": np.median(GS_feature),
                    "GSlower": np.percentile(GS_feature, 25),
                    "GSupper": np.percentile(GS_feature, 75),
                    "p": ranksums(ID_feature, GS_feature).pvalue,
                }
            )
        df_features = pd.DataFrame(
            rows, columns=["feature", "IDmedian", "IDlower", "IDupper", "GSmedian", "GSlower", "GSupper", "p"]
        )
        df_features.to_csv(os.path.join(save_path, f"features_{phase}.csv"), index=False, float_format="%.4f")

    # Features difference between phases
    phases_combinations = list(combinations(phases, 2))
    for phase1, phase2 in phases_combinations:
        rows = []
        for feature in features["ID"][phase1].keys():
            ID_diff = features["ID"][phase2][feature] - features["ID"][phase1][feature]
            GS_diff = features["GS"][phase2][feature] - features["GS"][phase1][feature]
            ID_diff = ID_diff[~np.isnan(ID_diff)]
            GS_diff = GS_diff[~np.isnan(GS_diff)]
            rows.append(
                {
                    "feature": f"{symbols[feature]} ({units[feature]})",
                    "IDmedian": np.median(ID_diff),
                    "IDlower": np.percentile(ID_diff, 25),
                    "IDupper": np.percentile(ID_diff, 75),
                    "IDp": wilcoxon(ID_diff).pvalue,
                    "GSmedian": np.median(GS_diff),
                    "GSlower": np.percentile(GS_diff, 25),
                    "GSupper": np.percentile(GS_diff, 75),
                    "GSp": wilcoxon(GS_diff).pvalue,
                    "p": ranksums(ID_diff, GS_diff).pvalue,
                }
            )
        df_features = pd.DataFrame(rows)
        df_features.to_csv(os.path.join(save_path, f"features_{phase1}_{phase2}.csv"), index=False, float_format="%.4f")

    n_subjects_diff = {}
    for phase1, phase2 in phases_combinations:
        n_subjects_diff[f"ID{phase1}{phase2}"] = (
            ~np.isnan(features["ID"][phase1][feature]) & ~np.isnan(features["ID"][phase2][feature])
        ).sum()
        n_subjects_diff[f"GS{phase1}{phase2}"] = (
            ~np.isnan(features["GS"][phase1][feature]) & ~np.isnan(features["GS"][phase2][feature])
        ).sum()
    df_n_subjects_diff = pd.DataFrame(n_subjects_diff, index=[0])
    df_n_subjects_diff.to_csv(os.path.join(save_path, "n_subjects_diff.csv"), index=False)
