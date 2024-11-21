from re import M
from typing import List, Dict
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_default_regions_names(n_regions: int) -> List[str]:
    return [f"M{m}" for m in range(1, n_regions + 1)]


def get_default_condition_names(n_conditions: int) -> List[str]:
    return [f"C{m}" for m in range(1, n_conditions + 1)]


def plot_coupling_estimate(
    A: np.ndarray,
    A_ground_truth: np.ndarray | None = None,
    lims=[-1.0, 1.0],
    region_names: List[str] | None = None,
    condition_names: List[str] | None = None,
    column_names: List[str] | None = None,
    cmap="seismic",
    title: str | None = None,
    save_to: str | None = None,
) -> None:
    """Plot directional matrix estimate against ground truth

    Args:
        A (np.ndarray): shape (n_conditions,n_regions,n_regions)
        A_ground_truth (np.ndarray | None, optional):
            shape (n_conditions,n_regions,n_regions). Defaults to None.
        lims (list, optional): Defaults to [-1.0, 1.0].
        region_names (List[str] | None, optional): Defaults to None.
        condition_names (List[str] | None, optional): Defaults to None.
        column_names (List[str] | None, optional): Defaults to None.
        cmap (str, optional): Defaults to "seismic".
        title (str | None, optional): Defaults to None.
        save_to (str | None, optional): filepath. Defaults to None.
    """

    n_conditions = A.shape[-3]
    n_regions = A.shape[-1]

    if region_names is None:
        region_names = get_default_regions_names(n_regions)
    if condition_names is None:
        condition_names = get_default_condition_names(n_conditions)

    fig, axes = plt.subplots(
        ncols=(1 if A_ground_truth is None else 2),
        nrows=n_conditions,
        figsize=(10, 10 * n_conditions),
        squeeze=False,
    )

    for c in range(n_conditions):
        im = axes[c, 0].imshow(A[c], vmin=lims[0], vmax=lims[1], cmap=cmap)
        if A_ground_truth is not None:
            axes[c, 1].imshow(
                A_ground_truth[c], vmin=lims[0], vmax=lims[1], cmap=cmap
            )

    for _, ax in np.ndenumerate(axes):
        ticks = [m for m in range(n_regions)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{m} -->" for m in region_names],
            rotation=90,
            ha="center",
            fontsize=15,
        )
        ax.set_yticks(ticks)

    for c, condition_name in enumerate(condition_names):
        axes[c, 0].set_ylabel(condition_name, fontsize=15)
        axes[c, 0].set_yticklabels(
            [f"{m} <--" for m in region_names],
            rotation=0,
            ha="right",
            fontsize=15,
        )
        if A_ground_truth is not None:
            axes[c, 1].yaxis.set_visible(False)

    if column_names is not None:
        for col, column_name in enumerate(column_names):
            axes[0, col].set_title(column_name, fontsize=15)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    if title is not None:
        plt.suptitle(title, fontsize=20)

    if save_to is not None:
        plt.savefig(
            save_to,
            bbox_inches="tight",
        )
    else:
        plt.show()


def plot_coupling_posterior(
    A_samples: np.ndarray,
    A_ground_truth: np.ndarray | None = None,
    lims=[-1.5, 1.5],
    region_names: List[str] | None = None,
    condition_names: List[str] | None = None,
) -> None:
    """Plot distribution of directional coupling estimates

    Args:
        A_samples (np.ndarray): shape (n_samples, n_conditions,n_regions,n_regions)
        A_ground_truth (np.ndarray | None, optional):
            shape (n_conditions,n_regions,n_regions). Defaults to None.
        lims (list, optional): Defaults to [-1.5, 1.5].
        region_names (List[str] | None, optional): Defaults to None.
        condition_names (List[str] | None, optional): Defaults to None.
    """

    n_regions = A_samples.shape[-1]
    n_conditions = A_samples.shape[-3]

    if region_names is None:
        region_names = get_default_regions_names(n_regions)
    if condition_names is None:
        condition_names = get_default_condition_names(n_conditions)

    _, axes = plt.subplots(
        n_regions,
        n_regions,
        figsize=(n_regions * 5, n_regions * 5),
        sharex="all",
        sharey="all",
    )
    for m1, region1 in enumerate(region_names):
        axes[0, m1].set_title(f"Going from {region1}", fontsize=15)
        axes[m1, 0].set_ylabel(f"Going to {region1}", fontsize=15)
        for m2, _ in enumerate(region_names):
            a_samples = A_samples[..., m1, m2]
            for c, condition in enumerate(condition_names):
                sns.kdeplot(
                    a_samples[:, c],
                    ax=axes[m1, m2],
                    label=f"Posterior samples\n{condition}",
                    color=f"C{c}",
                )
                if A_ground_truth is not None:
                    a = A_ground_truth[c, m1, m2]
                    axes[m1, m2].axvline(
                        a,
                        color=f"C{c}",
                        alpha=0.5,
                        label=f"Ground truth\n{condition}",
                    )
            axes[m1, m2].set_yticks([])
            axes[m1, m2].set_xticks([-1.0, 0.0, 1.0])
            axes[m1, m2].set_xticklabels([-1, 0, 1], fontsize=15)

    axes[0, 0].legend(fontsize=15)
    axes[0, 0].set_xlim(lims)
    plt.suptitle("Matrix A (directional coupling)\ninference", fontsize=20)
    plt.show()


def plot_signal_estimate(
    signal: np.ndarray,
    tr: float = 1.0,
    region_names: List[str] | None = None,
) -> None:
    """Plot latent/BOLD signal

    Args:
        signal (np.ndarray): shape (n_regions, n_timesteps)
        tr (float, optional): Defaults to 1.0.
        region_names (List[str] | None, optional): Defaults to None.
    """

    n_regions = signal.shape[-2]
    duration = signal.shape[-1]

    if region_names is None:
        region_names = get_default_regions_names(n_regions)

    _, axes = plt.subplots(
        n_regions, 1, figsize=(10, n_regions * 5), sharex="all", sharey="col"
    )
    axes = axes.reshape((n_regions, 1))
    for m, region in enumerate(region_names):
        t_range = np.arange(0.0, duration * tr, tr)
        axes[m, 0].plot(
            t_range,
            signal[m, :],
            "--",
            color=f"C{m}",
            label="Observed",
        )
        axes[m, 0].set_ylabel(f"Region {region}")

    axes[0, 0].set_title("Signal (observed)")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[0, 0].legend()


def plot_signal_posterior(
    signal_samples: np.ndarray,
    signal_ground_truth: np.ndarray | None = None,
    percentile_1: int = 1,
    percentile_2: int = 99,
    xlim: List[float] | None = None,
    ylim: List[float] | None = [-6.0, 6.0],
    tr: float = 1.0,
    region_names: List[str] | None = None,
) -> None:
    """Plot posterior of latent signal

    Args:
        signal_samples (np.ndarray): shape (n_samples, n_regions, n_timesteps)
        signal_ground_truth (np.ndarray | None, optional):
            shape (n_regions, n_timesteps) Defaults to None.
        percentile_1 (int, optional): for shaded regions Defaults to 1.
        percentile_2 (int, optional): for shaded region Defaults to 99.
        xlim (List[float] | None, optional): Defaults to None.
        ylim (List[float] | None, optional): Defaults to [-6.0, 6.0].
        tr (float, optional): Defaults to 1.0.
        region_names (List[str] | None, optional): Defaults to None.
    """

    n_regions = signal_samples.shape[-2]
    duration = signal_samples.shape[-1]

    if region_names is None:
        region_names = get_default_regions_names(n_regions)

    _, axes = plt.subplots(
        n_regions, 1, figsize=(10, n_regions * 5), sharex="all", sharey="col"
    )
    axes = axes.reshape((n_regions, 1))
    for m, region in enumerate(region_names):
        t_range = np.arange(0.0, duration * tr, tr)
        x_samples = signal_samples[..., m, :]
        axes[m, 0].plot(
            t_range,
            np.nanpercentile(x_samples, 50, axis=0),
            color=f"C{m}",
            label="Posterior sample median",
        )
        axes[m, 0].fill_between(
            t_range,
            np.nanpercentile(x_samples, percentile_1, axis=0),
            np.nanpercentile(x_samples, percentile_2, axis=0),
            color=f"C{m}",
            alpha=0.3,
            label=f"Posterior {percentile_2 - percentile_1:.0f}% confidence",
        )
        if signal_ground_truth is not None:
            axes[m, 0].plot(
                t_range,
                signal_ground_truth[m],
                "--",
                color=f"C{m}",
                label="Ground truth",
            )
        axes[m, 0].set_ylabel(f"Region {region}")
        if xlim is not None:
            axes[m, 0].set_xlim(xlim)
        if ylim is not None:
            axes[m, 0].set_ylim(ylim)

    axes[0, 0].set_title("Signal\n(infered vs ground truth)")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[0, 0].legend()


def plot_hyper_parameters_posterior(
    hyper_parameter_samples: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray] | None = None,
    region_names: List[str] | None = None,
    pairplot: bool = True,
) -> None:
    """Plot alpha (HRF angle), r, q (BOLD and latent-levels noise variances)

    Args:
        hyper_parameter_samples (Dict[str, np.ndarray]): dict(
            alpha=shape (n_samples, n_regions),
            r=shape (n_samples, n_regions),
            q=shape (n_samples, n_regions),
        )
        ground_truth (Dict[str, np.ndarray] | None, optional): Defaults to None.
        region_names (List[str] | None, optional): Defaults to None.
        pairplot (bool, optional): Defaults to True.
    """

    N_PARAMS = 3

    n_regions = hyper_parameter_samples["alpha"].shape[-1]
    if region_names is None:
        region_names = get_default_regions_names(n_regions)

    columns = (
        [f"alpha_{m}" for m in region_names]
        + [f"r_{m}" for m in region_names]
        + [f"q_{m}" for m in region_names]
    )
    estimates_df = pd.DataFrame(
        np.stack(
            [hyper_parameter_samples["alpha"][:, m] for m in range(n_regions)]
            + [hyper_parameter_samples["r"][:, m] for m in range(n_regions)]
            + [hyper_parameter_samples["q"][:, m] for m in range(n_regions)],
            axis=-1,
        ),
        columns=columns,
    )
    if ground_truth is not None:
        ground_truth_df = pd.DataFrame(
            [
                [ground_truth["b_angle"][m] for m in range(n_regions)]
                + [ground_truth["R"][m] for m in range(n_regions)]
                + [ground_truth["Q"][m] for m in range(n_regions)]
            ],
            columns=columns,
        )
    else:
        ground_truth_df = None

    if pairplot:
        grid = sns.pairplot(
            estimates_df,
            kind="kde",
            corner=True,
        )
        if ground_truth_df is not None:
            ground_truth_values = ground_truth_df.values[0]
            for i1 in range(N_PARAMS):
                for i2 in range(0, i1 + 1):
                    if i1 == i2:
                        grid.axes[i1, i2].axvline(
                            ground_truth_values[i1], color="red", ls="--"
                        )
                    else:
                        grid.axes[i1, i2].axhline(
                            ground_truth_values[i1], color="red", ls="--"
                        )
                        grid.axes[i1, i2].axvline(
                            ground_truth_values[i2], color="red", ls="--"
                        )
    else:
        _, axes = plt.subplots(
            nrows=N_PARAMS, ncols=1, figsize=(5, N_PARAMS * 5)
        )
        for i1, col in enumerate(estimates_df.columns):
            sns.kdeplot(estimates_df.values[:, i1], ax=axes[i1])
            axes[i1].set_xlabel(col)
            if ground_truth_df is not None:
                axes[i1].axvline(
                    ground_truth_df.values[0][i1], color="red", ls="--"
                )

    plt.show()
