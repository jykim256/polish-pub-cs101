import sys

import numpy as np
import tensorflow as tf
from optparse import OptionParser

from data import RadioSky
from model.wdsr import wdsr_b, wdsr_b_uq
from train import WdsrTrainer


def main(
    images_dir,
    caches_dir,
    fnoutweights,
    ntrain=800,
    nvalid=100,
    scale=4,
    nchan=1,
    nbit=16,
    num_res_blocks=32,
    batchsize=4,
    train_steps=10000,
):
    print(
        "Note we are assuming the following model checkpoint:",
        f".ckpt/%s" % fnoutweights.strip(".h5"),
    )
    trainer = WdsrTrainer(
        model=wdsr_b_uq(scale=scale, num_res_blocks=num_res_blocks, nchan=nchan),
        checkpoint_dir=f".ckpt/%s" % fnoutweights.strip(".h5"),
    )
    print("Loaded in trainer")

    # Train WDSR B model for train_steps steps and evaluate model
    # every 1000 steps on the first 10 images of the DIV2K
    # validation set. Save a checkpoint only if evaluation
    # PSNR has improved.

    print("Evaluating...")

    trainer.restore()
    # Evaluate model on full validation set.
    # psnr = trainer.evaluate(valid_ds)
    # print(f"PSNR = {psnr.numpy():3f}")

    # Save weights to separate location.
    trainer.model.save_weights(fnoutweights)


if __name__ == "__main__":
    parser = OptionParser(
        prog="train_model",
        version="",
        usage="%prog fname datestr specnum [OPTIONS]",
        description="Visualize and classify filterbank data",
    )
    parser.add_option(
        "-c",
        "--cachdir",
        dest="caches_dir",
        default=None,
        help="directory with training/validation image data",
    )
    parser.add_option(
        "-f",
        "--fnout",
        dest="fnout_model",
        type=str,
        default="model.h5",
        help="directory with training/validation image data",
    )
    parser.add_option(
        "-r", "--scale", dest="scale", type=int, default=4, help="upsample factor"
    )
    parser.add_option(
        "--nchan",
        dest="nchan",
        type=int,
        default=1,
        help="number of frequency channels in images",
    )
    parser.add_option(
        "--num_res_blocks",
        dest="num_res_blocks",
        type=int,
        default=32,
        help="number of residual blocks in neural network",
    )
    parser.add_option(
        "--nbit", dest="nbit", type=int, default=16, help="number of bits in image data"
    )
    parser.add_option(
        "--train_steps",
        dest="train_steps",
        type=int,
        default=100000,
        help="number of training steps",
    )
    parser.add_option(
        "--ntrain",
        dest="ntrain",
        type=int,
        help="number of training images",
        default=800,
    )
    parser.add_option(
        "--nvalid",
        dest="nvalid",
        type=int,
        help="number of validation images",
        default=100,
    )

    options, args = parser.parse_args()
    images_dir = args[0]

    if options.caches_dir is None:
        if images_dir[-1] == "/":
            caches_dir = images_dir[:1] + "-cache"
        else:
            caches_dir = images_dir + "-cache"
    else:
        caches_dir = options.caches_dir

    main(
        images_dir,
        caches_dir,
        options.fnout_model,
        ntrain=options.ntrain,
        nvalid=options.nvalid,
        scale=options.scale,
        nchan=options.nchan,
        nbit=options.nbit,
        num_res_blocks=options.num_res_blocks,
        train_steps=options.train_steps,
    )