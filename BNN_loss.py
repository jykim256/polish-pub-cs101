import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def laplacian_normalized_noexp(y_pred, y_true, show_parts=True):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    scale_pred = K.pow(y_pred[:, :, :, 1], 2) + 1e-7
    if show_parts:
        tf.print("top_loss", tf.math.reduce_mean(K.abs(mean_true - mean_pred)))
        tf.print("bottom_loss", tf.math.reduce_mean(scale_pred))
        tf.print("coef_loss", tf.math.reduce_mean(K.log(scale_pred)))
    loss = tf.math.divide((K.abs(mean_true - mean_pred)), scale_pred) + K.log(
        scale_pred
    )
    return loss


def laplacian_denormalized_noexp(y_pred, y_true, show_parts=True):
    mean_true = tf.math.scalar_mul(2**16, y_true[:, :, :, 0])
    mean_pred = tf.math.scalar_mul(2**16, y_pred[:, :, :, 0])
    scale_pred = tf.math.scalar_mul(2**16, y_pred[:, :, :, 1])
    if show_parts:
        tf.print("top_loss", tf.math.reduce_mean(K.abs(mean_true - mean_pred)))
        tf.print("bottom_loss", tf.math.reduce_mean(scale_pred))
        tf.print("coef_loss", tf.math.reduce_mean(K.log(scale_pred)))
    loss = tf.math.divide((K.abs(mean_true - mean_pred)), scale_pred) + K.log(
        scale_pred
    )
    return loss


def gaussian_normalized_exp(y_pred, y_true, show_parts=True):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    scale_pred = y_pred[:, :, :, 1]
    if show_parts:
        tf.print("sqrt top_loss", tf.math.reduce_mean(K.abs(mean_true - mean_pred)))
        tf.print("bottom_loss", tf.math.reduce_mean(2 * K.exp(scale_pred)))
        tf.print("coef_loss", (tf.math.reduce_mean(scale_pred) / 2))
    loss = tf.math.divide((K.pow(mean_true - mean_pred, 2)), 2 * K.exp(scale_pred)) + (
        tf.divide(scale_pred, 2)
    )
    return loss


def gaussian_denormalized_exp(y_pred, y_true, show_parts=True):
    mean_true = tf.math.scalar_mul(2**16, y_true[:, :, :, 0])
    mean_pred = tf.math.scalar_mul(2**16, y_pred[:, :, :, 0])
    scale_pred = tf.math.scalar_mul(2**16, y_pred[:, :, :, 1])
    if show_parts:
        tf.print("sqrt top_loss", tf.math.reduce_mean(K.abs(mean_true - mean_pred)))
        tf.print("bottom_loss", tf.math.reduce_mean(2 * K.exp(scale_pred)))
        tf.print("coef_loss", (tf.math.reduce_mean(scale_pred) / 2))
    loss = tf.math.divide((K.pow(mean_true - mean_pred, 2)), 2 * K.exp(scale_pred)) + (
        tf.divide(scale_pred, 2)
    )
    return loss


def gaussian_denormalized_noexp(y_pred, y_true, show_parts=True):
    mean_true = tf.math.scalar_mul(2**16, y_true[:, :, :, 0])
    mean_pred = tf.math.scalar_mul(2**16, y_pred[:, :, :, 0])
    scale_pred = K.pow(tf.math.scalar_mul(2**16, y_pred[:, :, :, 1]), 2) + 1e-7
    if show_parts:
        tf.print("top_loss", tf.math.reduce_mean(K.abs(mean_true - mean_pred)))
        tf.print("bottom_loss", tf.math.reduce_mean((scale_pred)))
        tf.print("coef_loss", (tf.math.reduce_mean(K.log(scale_pred))))
    loss = tf.math.divide((K.pow(mean_true - mean_pred, 2)), scale_pred) + (
        K.log(scale_pred)
    )
    return loss


def gaussian_normalized_noexp(y_pred, y_true, show_parts=True):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    scale_pred = K.pow(y_pred[:, :, :, 1], 2)
    if show_parts:
        tf.print("top_loss", tf.math.reduce_mean(K.abs(mean_true - mean_pred)))
        tf.print("bottom_loss", tf.math.reduce_mean((scale_pred)))
        tf.print("coef_loss", (tf.math.reduce_mean(K.log(scale_pred))))
    loss = tf.math.divide((K.pow(mean_true - mean_pred, 2)), scale_pred) + (
        K.log(scale_pred)
    )
    return loss


def gaussian_loss_non_exp(y_pred, y_true):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    scale_pred = K.pow(y_pred[:, :, :, 1], 2)
    loss = tf.math.divide((K.pow(mean_true - mean_pred, 2)), scale_pred) + K.log(
        scale_pred
    )
    return loss
