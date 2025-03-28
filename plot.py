import os

import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["cm"]
plt.rcParams["font.size"] = 12


def create_subplots(nrows) -> tuple[plt.Figure, list[plt.Axes]]:
    # Subplot size
    width, height = 5, 2  # inches

    # Figure margins
    left, right = 0.7, 0.7  # inches
    top, bottom = 0.3, 0.6  # inches

    # Vertical space before subplots
    vspace = 0.2  # inches

    # Create figure with margins
    fig_height = nrows * height + top + bottom  # inches
    fig_width = width + left + right  # inches
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create axes with shared x
    axes = []
    for i in range(nrows):
        ax_left = left / fig_width  # fraction
        ax_bottom = (fig_height - top - (i + 1) * height) / fig_height  # fraction
        ax_width = (fig_width - right - left) / fig_width  # fraction
        ax_height = (height - vspace) / fig_height  # fraction
        sharex = axes[0] if axes else None
        ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height], sharex=sharex)
        if i < nrows - 1:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    return fig, axes


def plot_results(title, save_path, tau_r, tau_f, tau_s, t, y_obs, phasic, tonic, u, u_obs=None):
    fig, ax = create_subplots(2)

    # Set title
    ax[0].set_title(f"{title}\n" + rf"($\tau_r = {tau_r:.2f}$, $\tau_f = {tau_f:.2f}$, $\tau_s = {tau_s:.2f}$)")

    # Plot y_obs, y_pred, and y_tonic
    ax[0].plot(t, y_obs, label=r"$y(t)$", color="black", linestyle=" ", marker=".")
    ax[0].plot(t, phasic + tonic, label=r"$y_T(t)+y_P(t)$", color="orange", linestyle="-")
    ax[0].plot(t, tonic, label=r"$y_T(t)$", color="blue", linestyle="-")
    ax[0].set_ylim(bottom=0, top=1.1 * max(y_obs))
    ax[0].set_ylabel(r"SC ($\mu S$)")
    ax[0].grid()
    ax[0].legend(loc="lower right")

    # Plot y_phasic
    ax[1].plot(t, phasic, label=r"$y_P(t)$", color="blue", linestyle="-")
    ax[1].set_ylim(bottom=0, top=1.1 * max(phasic))
    ax[1].set_ylabel(r"Phasic SC ($\mu S$)")
    ax[1].grid()
    # Plot u
    ax1 = ax[1].twinx()
    if u_obs is not None:
        ax1.vlines(t, 0, (u_obs > 0) * 1.1 * max(u), label=r"CS", color="gray", linewidth=2, alpha=0.5)
    ax1.stem(t, u, label=r"$u(t)$", linefmt="red", basefmt=" ", markerfmt=" ")
    ax1.set_ylim(bottom=0, top=max(1.1 * max(u), 1))
    ax1.set_ylabel(r"ANS Activations ($\mu S/s$)")
    lines, labels = ax[1].get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    # Set x-axis
    ax[1].set_xlim(left=0, right=t[-1])
    ax[1].set_xlabel(r"Time ($s$)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_activations(title, save_path, tau_r, tau_f, tau_s, t, y_obs, u):
    fig, axes = create_subplots(1)
    ax = axes[0]

    # Set title
    ax.set_title(f"{title}\n" + rf"($\tau_r = {tau_r:.2f}$, $\tau_f = {tau_f:.2f}$, $\tau_s = {tau_s:.2f}$)")

    # Plot y_obs
    ax.plot(t, y_obs, label=r"$y(t)$", color="black", linestyle=" ", marker=".")
    ax.set_ylim(bottom=0, top=1.1 * max(y_obs))
    ax.set_ylabel(r"SC ($\mu S$)")
    ax.grid()

    # Plot u
    ax1 = ax.twinx()
    ax1.stem(t, u, label=r"$u(t)$", linefmt="red", basefmt=" ", markerfmt=" ")
    ax1.set_ylim(bottom=0, top=max(1.1 * max(u), 1))
    ax1.set_ylabel(r"ANS Activations ($\mu S/s$)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    # Set x-axis
    ax.set_xlim(left=0, right=t[-1])
    ax.set_xlabel(r"Time ($s$)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    from scipy.io import loadmat

    exp_name = "best"
    load_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"

    # Plot full deconvolution results
    save_path = "results/paper/plots/{filename}_deconvolution.pdf"
    subject_phase_list = [
        ("4IIAPOJ_189", "cond", r"\textbf{ID} Participant 1 in \textbf{Conditioning} Phase", "id_1_cond")
    ]
    for subject, phase, title, filename in subject_phase_list:
        # Load result
        _load_path = load_path.format(subject=subject, phase=phase)
        result = loadmat(_load_path)
        tau_r = result["tau_r"].item()
        tau_f = result["tau_f"].item()
        tau_s = result["tau_s"].item()
        t = result["t"].flatten()
        y_obs = result["y_obs"].flatten()
        phasic = result["phasic"].flatten()
        tonic = result["tonic"].flatten()
        u = result["u"].flatten()

        # Plot
        _save_path = save_path.format(filename=filename)
        plot_results(title, _save_path, tau_r, tau_f, tau_s, t, y_obs, phasic, tonic, u)

    # Plot activations only
    save_path_u = "results/paper/plots/{filename}_activations.pdf"
    subject_phase_list = [
        ("4IIAPOJ_189", "cond", r"\textbf{ID} Participant 1 in \textbf{Conditioning} Phase", "id_1_cond"),
        ("4IIAPOJ_189", "ext", r"\textbf{ID} Participant 1 in \textbf{Extinction} Phase", "id_1_ext"),
        ("4IIAPOJ_189", "recall", r"\textbf{ID} Participant 1 in \textbf{Recall} Phase", "id_1_recall"),
        ("2INISAM_194", "cond", r"\textbf{GS} Participant 1 in \textbf{Conditioning} Phase", "gs_1_cond"),
        ("2INISAM_194", "ext", r"\textbf{GS} Participant 1 in \textbf{Extinction} Phase", "gs_1_ext"),
        ("2INISAM_194", "recall", r"\textbf{GS} Participant 1 in \textbf{Recall} Phase", "gs_1_recall"),
    ]
    for subject, phase, title, filename in subject_phase_list:
        # Load result
        _load_path = load_path.format(subject=subject, phase=phase)
        result = loadmat(_load_path)
        tau_r = result["tau_r"].item()
        tau_f = result["tau_f"].item()
        tau_s = result["tau_s"].item()
        t = result["t"].flatten()
        y_obs = result["y_obs"].flatten()
        u = result["u"].flatten()

        # Plot
        _save_path_u = save_path_u.format(filename=filename)
        plot_activations(title, _save_path_u, tau_r, tau_f, tau_s, t, y_obs, u)
