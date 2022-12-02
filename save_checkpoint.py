import sys

import numpy as np
import tensorflow as tf
from optparse import OptionParser

from data import RadioSky
from model.wdsr import wdsr_b, wdsr_b_uq
from train import WdsrTrainer


def save_model(
    fnoutweights,
    scale=4,
    nchan=1,
    num_res_blocks=32,
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
    print(trainer.checkpoint_manager.checkpoints)
    trainer.restore()
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
    options, args = parser.parse_args()

    save_model(
        options.fnout_model,
        scale=options.scale,
        nchan=options.nchan,
        num_res_blocks=options.num_res_blocks,
    )
