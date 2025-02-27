import os

import matplotlib.pyplot as plt
from scipy.io import loadmat

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["cm"]
plt.rcParams["font.size"] = 12


def plot_results(title, _save_path, t, y_obs, phasic, tonic, u, u_obs, tau_r, tau_f, tau_s):
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    fig.suptitle(f"{title}\n($\\tau_r = {tau_r:.2f}$, $\\tau_f = {tau_f:.2f}$, $\\tau_s = {tau_s:.2f}$)")

    # Plot y_obs, y_pred, and y_tonic
    ax[0].plot(t, y_obs, label="$y(t)$", color="black", linestyle=" ", marker=".")
    ax[0].plot(t, phasic + tonic, label="$y_T(t)+y_P(t)$", color="orange", linestyle="-")
    ax[0].plot(t, tonic, label="$y_T(t)$", color="blue", linestyle="-")
    ax[0].set_ylim(bottom=0, top=1.1 * max(y_obs))
    ax[0].set_ylabel("SC ($\\mu$S)")
    ax[0].grid()
    ax[0].legend(loc="lower right")

    # Plot y_phasic
    ax[1].plot(t, phasic, label="$y_P(t)$", color="blue", linestyle="-")
    ax[1].set_ylim(bottom=0, top=1.1 * max(phasic))
    ax[1].set_ylabel("Phasic SC ($\\mu$S)")
    ax[1].grid()

    # Plot u
    ax1 = ax[1].twinx()
    ax1.vlines(t, 0, (u_obs > 0) * 1.1 * max(u), label="CS", color="gray", linewidth=2, alpha=0.5)
    ax1.stem(t, u, label="$u(t)$", linefmt="red", basefmt=" ", markerfmt=" ")
    ax1.set_ylim(bottom=0, top=1.1 * max(u))
    ax1.set_ylabel("ANS Activations ($\\mu$S/s)")
    ax1.set_xlim(left=0, right=t[-1])
    ax1.set_xlabel("Time (s)")

    lines, labels = ax[1].get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()

    os.makedirs(os.path.dirname(_save_path), exist_ok=True)
    fig.savefig(_save_path)
    plt.close(fig)


def plot_activations(title, _save_path, t, y_obs, u, tau_r, tau_f, tau_s):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharex=True)
    fig.suptitle(f"{title}\n($\\tau_r = {tau_r:.2f}$, $\\tau_f = {tau_f:.2f}$, $\\tau_s = {tau_s:.2f}$)")

    # Plot y_obs
    ax.plot(t, y_obs, label="$y(t)$", color="black", linestyle=" ", marker=".")
    ax.set_ylim(bottom=0, top=1.1 * max(y_obs))
    ax.set_ylabel("SC ($\\mu$S)")
    ax.grid()

    # Plot u
    ax1 = ax.twinx()
    ax1.stem(t, u, label="$u(t)$", linefmt="red", basefmt=" ", markerfmt=" ")
    ax1.set_ylim(bottom=0, top=1.1 * max(u))
    ax1.set_ylabel("ANS Activations ($\\mu$S/s)")
    ax1.set_xlim(left=0, right=t[-1])
    ax1.set_xlabel("Time (s)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()

    os.makedirs(os.path.dirname(_save_path), exist_ok=True)
    fig.savefig(_save_path)
    plt.close(fig)


if __name__ == "__main__":
    exp_name = "best"
    load_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"

    # Plot full deconvolution results
    save_path = "results/plots/paper/{filename}_deconvolution.pdf"
    subject_phase_list = [
        ("4IIAPOJ_189", "cond", "\\textbf{ID} Participant 1 in \\textbf{Conditioning} Phase", "id_1_cond")
    ]
    for subject, phase, title, filename in subject_phase_list:
        # Load result
        _load_path = load_path.format(subject=subject, phase=phase)
        result = loadmat(_load_path)
        t = result["t"].flatten()
        y_obs = result["y_obs"].flatten()
        phasic = result["phasic"].flatten()
        tonic = result["tonic"].flatten()
        u = result["u"].flatten()
        u_obs = result["u_obs"].flatten()
        tau_r = result["tau_r"].item()
        tau_f = result["tau_f"].item()
        tau_s = result["tau_s"].item()

        # Plot
        _save_path = save_path.format(filename=filename)
        plot_results(title, _save_path, t, y_obs, phasic, tonic, u, u_obs, tau_r, tau_f, tau_s)

    # Plot activations only
    save_path_u = "results/plots/paper/{filename}_activations.pdf"
    subject_phase_list = [
        ("4IIAPOJ_189", "cond", "\\textbf{ID} Participant 1 in \\textbf{Conditioning} Phase", "id_1_cond"),
        ("4IIAPOJ_189", "ext", "\\textbf{ID} Participant 1 in \\textbf{Extinction} Phase", "id_1_ext"),
        ("4IIAPOJ_189", "recall", "\\textbf{ID} Participant 1 in \\textbf{Recall} Phase", "id_1_recall"),
        ("2INISAM_194", "cond", "\\textbf{GS} Participant 1 in \\textbf{Conditioning} Phase", "gs_1_cond"),
        ("2INISAM_194", "ext", "\\textbf{GS} Participant 1 in \\textbf{Extinction} Phase", "gs_1_ext"),
        ("2INISAM_194", "recall", "\\textbf{GS} Participant 1 in \\textbf{Recall} Phase", "gs_1_recall"),
    ]
    for subject, phase, title, filename in subject_phase_list:
        # Load result
        _load_path = load_path.format(subject=subject, phase=phase)
        result = loadmat(_load_path)
        t = result["t"].flatten()
        y_obs = result["y_obs"].flatten()
        u = result["u"].flatten()
        tau_r = result["tau_r"].item()
        tau_f = result["tau_f"].item()
        tau_s = result["tau_s"].item()

        # Plot
        _save_path_u = save_path_u.format(filename=filename)
        plot_activations(title, _save_path_u, t, y_obs, u, tau_r, tau_f, tau_s)
