import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["cm"]
plt.rcParams["font.size"] = 12


def create_subplots(nrows) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create subplots with specific size.
    """

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


def save_figure(fig: plt.Figure, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)


def combine_legends(*axes: plt.Axes, loc="upper right"):
    all_lines, all_labels = [], []
    for ax in axes:
        lines, labels = ax.get_legend_handles_labels()
        all_lines.extend(lines)
        all_labels.extend(labels)
        ax.legend().remove()
    axes[-1].legend(all_lines, all_labels, loc=loc)


def twinx(plot):
    def _plot(ax: plt.Axes, *args, **kwargs):
        ax1 = ax.twinx()
        plot(ax1, *args, **kwargs)
        ax1.grid(False)
        ax1.set_title("")
        combine_legends(ax, ax1)
        return ax1

    return _plot


def plot_measurements(ax: plt.Axes, t, y_obs, phasic=None, tonic=None):
    ax.plot(t, y_obs, label=r"$y$", color="black", linestyle=" ", marker=".")
    if tonic is not None:
        if phasic is not None:
            ax.plot(t, phasic + tonic, label=r"$y_T+y_P$", color="orange", linestyle="-")
        ax.plot(t, tonic, label=r"$y_T$", color="blue", linestyle="-")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"SC ($\mu S$)")
    ax.grid()
    ax.legend(loc="upper right")


def plot_phasic(ax: plt.Axes, t, phasic):
    ax.plot(t, phasic, label=r"$y_P$", color="blue", linestyle="-")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"Phasic SC ($\mu S$)")
    ax.grid()
    ax.legend(loc="upper right")


def plot_activations(ax: plt.Axes, t, u, u_obs=None):
    if u_obs is not None:
        ax.vlines(t, 0, (u_obs > 0) * 1.1 * max(u), label=r"CS", color="gray", linewidth=2, alpha=0.5)
    ax.stem(t, u, label=r"$u$", linefmt="red", basefmt=" ", markerfmt=" ")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"ANS Activation ($\mu S/s$)")
    ax.grid()
    ax.legend(loc="upper right")


def plot_time(ax: plt.Axes, t):
    ax.set_xlim(left=t[0], right=t[-1])
    ax.set_xlabel(r"Time ($s$)")


def plot_title(ax: plt.Axes, tau_r, tau_f, tau_s, title=None):
    _title = rf"($\tau_r = {tau_r:.2f}$, $\tau_f = {tau_f:.2f}$, $\tau_s = {tau_s:.2f}$)"
    if title is not None:
        _title = f"{title}\n" + _title
    ax.set_title(_title)


def plot_results(t, y_obs, phasic, tonic, u, tau_r, tau_f, tau_s, u_obs=None, title=None, save_path=None):
    fig, axes = create_subplots(2)

    plot_measurements(axes[0], t, y_obs, phasic=phasic, tonic=tonic)
    plot_phasic(axes[1], t, phasic)
    twinx(plot_activations)(axes[1], t, u, u_obs=u_obs)
    plot_time(axes[1], t)
    plot_title(axes[0], tau_r, tau_f, tau_s, title=title)

    if save_path is not None:
        save_figure(fig, save_path)
        plt.close(fig)
    else:
        return fig, axes


def plot_results_activations_only(t, y_obs, u, tau_r, tau_f, tau_s, title=None, save_path=None):
    fig, axes = create_subplots(1)
    ax = axes[0]

    plot_measurements(ax, t, y_obs)
    twinx(plot_activations)(ax, t, u)
    plot_time(ax, t)
    plot_title(ax, tau_r, tau_f, tau_s, title=title)

    if save_path is not None:
        save_figure(fig, save_path)
        plt.close(fig)
    else:
        return fig, axes


if __name__ == "__main__":
    exp_name = "best"
    load_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"
    save_path = f"results/paper/plots"

    os.makedirs(save_path, exist_ok=True)

    # Dictionary for display
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
    datalist = datalist[datalist["status"] == "included"]

    # Exclude over-stimulated subjects
    def over_stimulated(row):
        result = loadmat(load_path.format(subject=row["subject"], phase=row["phase"]))
        return np.sum(result["u_obs"] > 0) > 50

    datalist = datalist[~datalist.apply(over_stimulated, axis=1)]

    # Main loop
    for _, row in datalist.iterrows():
        subject_id = row["subject_id"]
        subject = row["subject"]
        phase = row["phase"]
        group = row["group"]

        # Load result
        _load_path = load_path.format(subject=subject, phase=phase)
        result = loadmat(_load_path)

        t = result["t"].squeeze(0)
        y_obs = result["y_obs"].squeeze(0)
        phasic = result["phasic"].squeeze(0)
        tonic = result["tonic"].squeeze(0)
        u = result["u"].squeeze(0)
        tau_r = result["tau_r"].item()
        tau_f = result["tau_f"].item()
        tau_s = result["tau_s"].item()

        # Plot
        phase_name = phase_names[phase]
        title = f"Subject {subject_id}, Group: \\textbf{{{group}}}, Phase: \\textbf{{{phase_name}}}"
        filename = f"{subject_id}_{phase}_{group.lower()}"

        _save_path = os.path.join(save_path, f"{filename}_deconvolution.pdf")
        plot_results(t, y_obs, phasic, tonic, u, tau_r, tau_f, tau_s, title=title, save_path=_save_path)

        _save_path = os.path.join(save_path, f"{filename}_activations.pdf")
        plot_results_activations_only(t, y_obs, u, tau_r, tau_f, tau_s, title=title, save_path=_save_path)
