import os
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ranksums, wilcoxon

if __name__ == "__main__":
    exp_name = "best"
    load_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"

    save_path = f"results/stats/{exp_name}"
    os.makedirs(save_path, exist_ok=True)

    # Get the list of subjects and phases
    datalist_path = "../data/datalist.csv"
    with open(datalist_path, "r") as f:
        datalist = pd.read_csv(f)
    datalist = datalist[datalist["group"].isin(["ID", "GS"])]

    # Get the list of phases
    phases = datalist["phase"].unique()

    # Get number of subjects for each phase and group
    rows = []
    for phase in phases:
        n_ID = ((datalist["phase"] == phase) & (datalist["group"] == "ID")).sum()
        n_GS = ((datalist["phase"] == phase) & (datalist["group"] == "GS")).sum()
        rows.append({"phase": phase, "ID": n_ID, "GS": n_GS})
    df_n_subjects = pd.DataFrame(rows, columns=["phase", "ID", "GS"])
    df_n_subjects.to_csv(os.path.join(save_path, "n_subjects.csv"), index=False)

    # Initialize dictionary {group: {phase: {feature: [value]}}}
    features = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Extract features
    for row in datalist.itertuples():
        subject = row.subject
        phase = row.phase
        group = row.group
        status = row.status

        if status == "included":
            # Load result
            _load_path = load_path.format(subject=subject, phase=phase)
            result = loadmat(_load_path)
        else:
            # Use nan for excluded data for pairwise tests
            result = defaultdict(lambda: np.array(np.nan))

        # Extract features
        features[group][phase]["tau_r"].append(result["tau_r"].item())
        features[group][phase]["tau_f"].append(result["tau_f"].item())
        features[group][phase]["tau_s"].append(result["tau_s"].item())
        features[group][phase]["u_0"].append(np.sum(result["u"] > 0).item())
        features[group][phase]["u_1"].append(np.linalg.vector_norm(result["u"], ord=1).item())
        features[group][phase]["u_2"].append(np.linalg.vector_norm(result["u"], ord=2).item())

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
            plt.boxplot([ID_feature, GS_feature], tick_labels=["ID", "GS"])
            plt.title(f"{phase} {feature}")
            plt.savefig(os.path.join(save_path, f"{feature}_{phase}.png"))
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
                    "feature": feature,
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
        df_features.to_csv(os.path.join(save_path, f"features_{phase}.csv"), index=False)

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
                    "feature": feature,
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
        df_features.to_csv(os.path.join(save_path, f"features_{phase1}_{phase2}.csv"), index=False)
