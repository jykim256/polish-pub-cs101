import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf


def plot_dictionary(data, cmap="afmhot", gamma=None, title='', num_columns = 3, interpolation=None):
    if interpolation is None:
        interpolation = 'antialiased'
    # data is in the following format
    # Dictionary['plot title', (Image, minbound, maxbound)], where Image is a 2D array
    num_plots = len(data)
    num_rows = int(np.ceil(num_plots / num_columns))
    fig = plt.figure(figsize=(6 * num_columns, 6 * num_rows))
    for plot_idx, (plot_title, (image_data_raw, minbound, maxbound)) in enumerate(
        data.items()
    ):
        if minbound == maxbound:
            minbound = np.min(image_data)
            maxbound = np.max(image_data)
        if gamma is not None:
            image_data = tf.image.adjust_gamma(tf.dtypes.cast(image_data_raw, tf.int32), gamma=gamma)
            minbound, maxbound = tuple(tf.image.adjust_gamma(np.array([float(minbound), float(maxbound)]), gamma=gamma).numpy())
        else:
            image_data = image_data_raw
        print(f" {plot_title} minbound = {minbound}, maxbound = {maxbound}")
        ax = plt.subplot(num_rows, num_columns, plot_idx + 1)
        plt.title(plot_title, c="k", fontsize=17)
        plt.imshow(
            tf.squeeze(tf.clip_by_value(image_data, minbound, maxbound)),
            vmin=minbound,
            vmax=maxbound,
            cmap=cmap,
            aspect="auto",
            extent=[0, 1, 0, 1],
            interpolation=interpolation
        )
        plt.axis("off")
        plt.colorbar()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle(title, fontsize=30)
    plt.show()



def plot_reconstruction(
    datalr, datasr, datahr=None, datauq=None, vm=1, nsub=2, cmap="afmhot"
):
    """Plot the dirty image, POLISH reconstruction,
    and (optionally) the high resolution true sky image
    """
    vminlr = 0
    vmaxlr = 22500
    vminsr = 0
    vmaxsr = 22500
    vminhr = 0
    vmaxhr = 22500
    if datahr is None:
        num_plots = 2
        fig = plt.figure(figsize=(10, 6))
    elif datauq is None:
        num_plots = 3
        fig = plt.figure(figsize=(16, 6))
    else:
        num_plots = 5
        fig = plt.figure(figsize=(22, 6))

    ax1 = plt.subplot(1, num_plots, 1)
    plt.title("Dirty map", color="C1", fontsize=17)
    plt.axis("off")
    plt.imshow(
        np.squeeze(datalr),
        cmap=cmap,
        aspect="auto",
        extent=[0, 1, 0, 1],
    )
    plt.setp(ax1.spines.values(), color="C1")

    ax2 = plt.subplot(1, num_plots, 2, sharex=ax1, sharey=ax1)
    plt.title("POLISH reconstruction", c="C2", fontsize=17)
    plt.imshow(
        np.squeeze(datasr),
        cmap=cmap,
        aspect="auto",
        extent=[0, 1, 0, 1],
    )
    plt.axis("off")

    if num_plots >= 3:
        ax3 = plt.subplot(1, num_plots, 3, sharex=ax1, sharey=ax1)
        plt.title("True sky", c="k", fontsize=17)
        plt.imshow(
            np.squeeze(datahr),
            cmap=cmap,
            aspect="auto",
            extent=[0, 1, 0, 1],
        )
        plt.axis("off")

    if num_plots >= 4:
        ax4 = plt.subplot(1, num_plots, 4, sharex=ax1, sharey=ax1)
        plt.title("Uncertainty", c="k", fontsize=17)
        plt.imshow(
            np.squeeze(datauq),
            cmap=cmap,
            aspect="auto",
            extent=[0, 1, 0, 1],
        )
        plt.axis("off")
        ax4 = plt.subplot(1, num_plots, 5, sharex=ax1, sharey=ax1)
        plt.title("Log Uncertainty", c="k", fontsize=17)
        plt.imshow(
            np.squeeze(np.log(datauq)),
            cmap=cmap,
            aspect="auto",
            extent=[0, 1, 0, 1],
        )
        plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
