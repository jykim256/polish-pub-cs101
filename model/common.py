import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from visualize import plot_dictionary, plot_reconstruction

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr, nbit=16):
    return resolve16(model, tf.expand_dims(lr, axis=0), nbit=nbit)[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float16)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def resolve16(model, lr_batch, nbit=16, get_raw=False):
    if nbit == 8:
        casttype = tf.uint8
    elif nbit == 16:
        casttype = tf.uint16
    else:
        print("Wrong number of bits")
        exit()
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_raw_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_raw_batch, 0, 2**nbit - 1)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, casttype)
    if get_raw:
        return sr_batch, sr_raw_batch
    return sr_batch


def evaluate(model, dataset, nbit=8, show_image=False, loss_name=""):
    # set datename to the current date and time in a string format
    datename = (
        model._name
        + "--"
        + loss_name
        + "--"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    psnr_values = []
    has_uq = "uq" in model._name
    lr_output, hr_output, sr_output, uq_output, sr_raw_output, uq_raw_output = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    for idx, (lr, hr) in enumerate(dataset):
        output, raw_output = resolve16(model, lr, nbit=nbit, get_raw=True)  # hack
        sr, uq = None, None
        sr_raw, uq_raw = None, None
        if has_uq:
            sr = tf.expand_dims(output[:, :, :, 0], -1)
            uq = tf.expand_dims(output[:, :, :, 1], -1)
            sr_raw = tf.expand_dims(raw_output[:, :, :, 0], -1)
            uq_raw = tf.expand_dims(raw_output[:, :, :, 1], -1)
        else:
            if lr.shape[-1] == 1:
                sr = output[..., 0, None]
        psnr_value = psnr(hr, sr, nbit=nbit)[0]
        psnr_values.append(psnr_value)
        # we only need to show one, just pick the first one
        if idx == 0:
            lr_output, hr_output, sr_output, uq_output = lr, hr, sr, uq
            sr_raw, uq_raw = sr_raw, uq_raw
    if show_image:
        plot_dictionary(
            {
                "Dirty Map":
                (lr_output, 0, 2**16),
                "Reconstruction Map":
                (denormalize(tf.dtypes.cast(sr_output, tf.float32)), 0, 2**16),
                "True Sky Map":
                (hr_output, 0, 2**16),
                "Uncertainty Map":
                (uq_output, 0, 2**16),
            },
            gamma=0.75,
        )

        # denormal_sr = denormalize(tf.dtypes.cast(sr_output, tf.float32))
        # gamma_sr = tf.image.adjust_gamma(denormal_sr, gamma=0.75)
        # # plot images here
        # plot_reconstruction(
        #     datalr=tf.image.adjust_gamma(lr_output, gamma=0.75),
        #     datahr=tf.image.adjust_gamma(hr_output, gamma=0.75),
        #     datasr=gamma_sr,
        #     datauq=uq_output,
        # )

        plt.hist(sr_raw.numpy().flatten(), bins=20)
        plt.yscale("log")
        plt.title("SR RAW histogram")
        fig = plt.gcf()
        fig.savefig(f"{datename}-srhist.png", dpi=300, format="png")
        plt.show()
        print("SR min/max: ", np.min(sr_raw.numpy()), np.max(sr_raw.numpy()))

        plt.hist(sr_output.numpy().flatten(), bins=20)
        plt.yscale("log")
        plt.title("SR histogram")
        fig = plt.gcf()
        fig.savefig(f"{datename}-srhist.png", dpi=300, format="png")
        plt.show()
        print("SR min/max: ", np.min(sr_output.numpy()), np.max(sr_output.numpy()))

        plt.hist(hr_output.numpy().flatten(), bins=20)
        plt.yscale("log")
        plt.title("HR histogram")
        fig = plt.gcf()
        fig.savefig(f"{datename}-hrhist.png", dpi=300, format="png")
        plt.show()
        print("HR min/max: ", np.min(hr_output.numpy()), np.max(hr_output.numpy()))

        if has_uq:

            plt.hist(uq_raw.numpy().flatten(), bins=20)
            plt.yscale("log")
            plt.title("Uncertainty RAW histogram")
            fig = plt.gcf()
            fig.savefig(f"{datename}-uqhist.png", dpi=300, format="png")
            plt.show()
            print(
                "UQ min/max: ",
                np.min(uq_raw.numpy()),
                np.max(uq_raw.numpy()),
            )

            plt.hist(uq_output.numpy().flatten(), bins=20)
            plt.yscale("log")
            plt.title("Uncertainty histogram")
            fig = plt.gcf()
            fig.savefig(f"{datename}-uqhist.png", dpi=300, format="png")
            plt.show()
            print("UQ min/max: ", np.min(uq_output.numpy()), np.max(uq_output.numpy()))
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------
# def normalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
#    if True:
#        return (x - rgb_mean) / 127.5
#    elif nbit==16:
#        return (x - 2.**15)/2.**15


# def denormalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
#    if True:
#        return x * 127.5 + rgb_mean


def normalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
    if nbit == 8:
        return (x - rgb_mean) / 127.5
    elif nbit == 16:
        return (x - 2.0**15) / 2.0**15


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
    if nbit == 8:
        return tf.dtypes.cast(x, tf.float32) * 127.5 + rgb_mean
    elif nbit == 16:
        return tf.dtypes.cast(x, tf.float32) * 2**15 + 2**15


def decenter(x):
    return x + 1


def decenternormalize(x):
    return x * 2**15


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2, nbit=8):
    return tf.image.psnr(x1, x2, max_val=2**nbit - 1)


def psnr16(x1, x2):
    return tf.image.psnr(x1, x2, max_val=2**16 - 1)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
