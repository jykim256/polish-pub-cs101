import optparse
import os
import sys
import time

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

from model import resolve16, resolve_single
from model.wdsr import *
from utils import load_image, plot_sample

vminlr = 0
vmaxlr = 22500
vminsr = 0
vmaxsr = 22500
vminhr = 0
vmaxhr = 22500

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 0.5,
        "lines.markersize": 5,
        "legend.fontsize": 14,
        "legend.borderaxespad": 0,
        "legend.frameon": False,
        "legend.loc": "lower right",
    }
)


def reconstruct(
    fn_img, fn_model, model_struct, iter, scale, fnhr=None, nbit=16, regular_image=False, dropout_rate = None
):
    if iter is None:
        iter = 1
    print("Loading image from ", fn_img)
    if fn_img.endswith("npy"):
        datalr = np.load(fn_img)[:, :]
    elif fn_img.endswith("png"):
        try:
            datalr = load_image(fn_img)
            if regular_image:
                # find maximum and minimum values of datalr
                # scale to the range 0 - vmaxlr
                # print("datalr min", np.min(datalr))
                # print("datalr max", np.max(datalr))
                # print('scale', ((vmaxlr) / (np.max(datalr) - np.min(datalr))))
                datalr = datalr * ((vmaxlr) / (np.max(datalr) - np.min(datalr)))
                # print("datalr min", np.min(datalr))
                # print("datalr max", np.max(datalr))
            # print('datalr shape', datalr.shape)
        except:
            return
    print("Loading HR image from ", fnhr)
    if fnhr is not None:
        if fnhr.endswith("npy"):
            datalr = np.load(fnhr)[:, :]
        elif fnhr.endswith("png"):
            try:
                datahr = load_image(fnhr)
            except:
                return
    else:
        datahr = None
    print("Loading model from ", fn_model)
    model = model_struct(scale=scale, num_res_blocks=32)
    if dropout_rate:
        model = model_struct(scale=scale, num_res_blocks=32, dropout_rate = dropout_rate)
    model.load_weights(fn_model)
    print("Model loaded")
    datalr = datalr[:, :, None]
    # print("datalrshape")
    # print(datalr.shape)
    # datalr = tf.stack([datalr, datalr], axis=3)

    if len(datalr.shape) == 4:
        # datalr = datalr.squeeze()
        datalr = datalr[:, :, :, 0]
    srs = []
    for idx in range(iter):
        print("Reconstructing image #%d" % idx)
        output, datasr = resolve16(
            model, tf.expand_dims(datalr, axis=0), nbit=nbit, get_raw=True
        )  # hack
        datasr = datasr.numpy()
        print("Range: %f - %f" % (np.min(datasr), np.max(datasr)))
        srs.append(datasr)

    datasr = np.array(srs)
    datasr = np.mean(datasr, axis=0)

    return datalr, datasr, datahr

    # def plot_reconstruction(
    #     datalr,
    #     datasr,
    #     datahr=None,
    #     vm=1,
    #     nsub=2,
    #     cmap="afmhot",
    #     regular_image=False,
    #     mc_data=None,
    # ):
    #     """Plot the dirty image, POLISH reconstruction,
    #     and (optionally) the high resolution true sky image
    #     """

    #     if nsub == 2:
    #         fig = plt.figure(figsize=(10, 6))
    #     if nsub == 3:
    #         fig = plt.figure(figsize=(13, 6))
    #     if mc_data is not None:
    #         fig = plt.figure(figsize=(16, 6))
    #     ax1 = plt.subplot(1, nsub, 1)
    #     plt.title("Dirty map", color="C1", fontsize=17)
    #     plt.axis("off")
    #     if regular_image:
    #         print("datalr shape", datalr.shape)
    #         plt.imshow(tf.squeeze(datalr), cmap="RdBu")
    #     else:
    #         plt.imshow(
    #             datalr[..., 0],
    #             cmap=cmap,
    #             vmax=vmaxlr,
    #             vmin=vminlr,
    #             aspect="auto",
    #             extent=[0, 1, 0, 1],
    #         )
    #     plt.setp(ax1.spines.values(), color="C1")

    #     ax2 = plt.subplot(1, nsub, 2, sharex=ax1, sharey=ax1)
    #     plt.title("POLISH reconstruction", c="C2", fontsize=17)
    #     if regular_image:
    #         print("datasr shape", datasr.shape)
    #         plt.imshow(tf.squeeze(datasr), cmap="RdBu")
    #     else:
    #         plt.imshow(
    #             tf.squeeze(datasr),
    #             cmap=cmap,
    #             vmax=vmaxsr,
    #             vmin=vminsr,
    #             aspect="auto",
    #             extent=[0, 1, 0, 1],
    #         )
    #     plt.axis("off")

    #     # print(np.sum(datahr))
    #     # print(np.sum(mc_data))

    #     ax3 = plt.subplot(1, nsub, 3, sharex=ax1, sharey=ax1)
    #     plt.title("True sky", c="k", fontsize=17)
    #     plt.imshow(
    #         tf.squeeze(datahr),
    #         cmap=cmap,
    #         vmax=vmaxsr,
    #         vmin=vminsr,
    #         aspect="auto",
    #         extent=[0, 1, 0, 1],
    #     )
    #     plt.axis("off")

    #     if mc_data is not None:
    #         ax4 = plt.subplot(1, nsub, 4, sharex=ax1, sharey=ax1)
    #         plt.title("Uncertainty", c="k", fontsize=17)
    #         plt.imshow(
    #             tf.squeeze(mc_data),
    #             cmap=cmap,
    #             vmax=vmaxsr,
    #             vmin=vminsr,
    #             aspect="auto",
    #             extent=[0, 1, 0, 1],
    #         )
    #         plt.axis("off")

    #     plt.tight_layout()
    #     plt.colorbar()
    #     plt.show()

    # def main(
    #     fn_img, fn_model, scale=4, fnhr=None, nbit=16, plotit=True, regular_image=False
    # ):
    #     datalr, datasr, datahr = reconstruct(
    #         fn_img, fn_model, scale, fnhr, nbit, regular_image=regular_image
    #     )
    #     if datahr is not None:
    #         nsub = 3
    #     else:
    #         nsub = 2
    #     print(datalr.shape)
    #     if plotit:
    #         plot_reconstruction(
    #             datalr,
    #             datasr[:, :, 0],
    #             datahr=datahr,
    #             vm=1,
    #             nsub=4,
    #             regular_image=regular_image,
    #             mc_data=datasr[:, :, 1],
    #         )

    # def main_mc_dropout(
    #     fn_img,
    #     fn_model,
    #     scale=4,
    #     fnhr=None,
    #     nbit=16,
    #     plotit=True,
    #     regular_image=False,
    #     num_iter=50,
    # ):
    #     datalr, datasr, datahr, mc_data = reconstruct_mc(
    #         fn_img,
    #         fn_model,
    #         scale,
    #         fnhr,
    #         nbit,
    #         regular_image=regular_image,
    #         num_iter=num_iter,
    #     )
    #     if datahr is not None:
    #         nsub = 3
    #     else:
    #         nsub = 2
    #     if plotit:
    #         mc_data = np.var(mc_data, axis=-1)
    #         plot_reconstruction(
    #             datalr,
    #             datasr,
    #             datahr=datahr,
    #             vm=1,
    #             nsub=4,
    #             regular_image=regular_image,
    #             mc_data=mc_data,
    #         )
    #     return mc_data

    # if __name__ == "__main__":
    # Example usage:
    # Generate images on training data:
    # for im in ./images/PSF-nkern64-4x/train/X4/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done
    # Generate images on validation data
    # for im in ./images/PSF-nkern64-4x/valid/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done

    parser = optparse.OptionParser(
        prog="hr2lr.py",
        version="",
        usage="%prog image weights.h5  [OPTIONS]",
        description="Take high resolution images, deconvolve them, \
                                   and save output.",
    )

    parser.add_option("-f", dest="fnhr", help="high-res file name", default=None)
    parser.add_option("-x", dest="scale", help="spatial rebin factor", default=4)
    parser.add_option(
        "-b",
        "--nbit",
        dest="nbit",
        type=int,
        help="number of bits in image",
        default=16,
    )
    parser.add_option("-p", "--plotit", dest="plotit", action="store_true", help="plot")

    options, args = parser.parse_args()
    fn_img, fn_model = args

    datalr, datasr, datahr = reconstruct(
        fn_img, fn_model, options.scale, fnhr=options.fnhr, nbit=options.nbit
    )

    if datahr is not None:
        nsub = 3
    else:
        nsub = 2

    if options.plotit:
        plot_reconstruction(datalr, datasr, datahr=datahr, vm=1, nsub=nsub)
