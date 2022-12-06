import optparse
import os
import sys

import matplotlib.pylab as plt

# from model.common import resolve_single
from model.wdsr import wdsr_b
from utils import load_image, plot_sample
from visualize import plot_reconstruction

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


def reconstruct(fn_img, fn_model, scale, fnhr=None, nbit=16, nchan=1):
    if fn_img.endswith("npy"):
        datalr = np.load(fn_img)[:, :]
    elif fn_img.endswith("png"):
        try:
            datalr = load_image(fn_img)
        except:
            return

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

    model = wdsr_b(scale=scale, num_res_blocks=32, nchan=nchan)
    model.load_weights(fn_model)

    if len(datalr.shape) == 3 and nchan == 1:
        datalr = datalr[:, :, 0, None]
    elif len(datalr.shape) == 2:
        datalr = datalr[:, :, None]

    print(datalr.shape)

    output, datasr = resolve16(model, tf.expand_dims(datalr, axis=0), nbit=nbit)[0]
    datasr = datasr.numpy()
    return datalr, datasr, datahr


if __name__ == "__main__":
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
        "-n",
        dest="nchan",
        type=int,
        help="number of frequency/color channels",
        default=1,
    )
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
        fn_img,
        fn_model,
        options.scale,
        fnhr=options.fnhr,
        nbit=options.nbit,
        nchan=options.nchan,
    )

    if datahr is not None:
        nsub = 3
    else:
        nsub = 2

    if options.plotit:
        plot_reconstruction(datalr, datasr, datahr=datahr, vm=1, nsub=nsub)
