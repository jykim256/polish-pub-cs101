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
    mean_true = tf.math.divide(y_true[:, :, :, 0], 2**16)
    mean_pred = tf.math.divide(y_pred[:, :, :, 0], 2**16)
    scale_pred = tf.math.divide(y_pred[:, :, :, 1], 2**16)
    if show_parts:
        tf.print("top_loss", tf.math.reduce_mean(K.abs(mean_true - mean_pred)))
        tf.print("bottom_loss", tf.math.reduce_mean(scale_pred))
        tf.print("coef_loss", tf.math.reduce_mean(K.log(scale_pred)))
    loss = tf.math.divide((K.abs(mean_true - mean_pred)), scale_pred) + K.log(
        scale_pred
    )
    return loss


def gaussian_normalized_exp(y_pred, y_true, show_parts=True):
    mean_true = tf.math.divide(y_true[:, :, :, 0], 2**16)
    mean_pred = tf.math.divide(y_pred[:, :, :, 0], 2**16)
    scale_pred = tf.math.divide(y_pred[:, :, :, 1], 2**16)
    if show_parts:
        tf.print("sqrt top_loss", tf.math.reduce_mean(K.abs(mean_true - mean_pred)))
        tf.print("bottom_loss", tf.math.reduce_mean(2 * K.exp(scale_pred)))
        tf.print("coef_loss", (tf.math.reduce_mean(scale_pred) / 2))
    loss = tf.math.divide((K.pow(mean_true - mean_pred, 2)), 2 * K.exp(scale_pred)) + (
        tf.divide(scale_pred, 2)
    )
    return loss


def gaussian_denormalized_exp(y_pred, y_true, show_parts=True):
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


def gaussian_denormalized_noexp(y_pred, y_true, show_parts=True):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    scale_pred = K.pow(y_pred[:, :, :, 1], 2) + 1e-7
    if show_parts:
        tf.print("top_loss", tf.math.reduce_mean(K.abs(mean_true - mean_pred)))
        tf.print("bottom_loss", tf.math.reduce_mean((scale_pred)))
        tf.print("coef_loss", (tf.math.reduce_mean(K.log(scale_pred))))
    loss = tf.math.divide((K.pow(mean_true - mean_pred, 2)), scale_pred) + (
        K.log(scale_pred)
    )
    return loss


def gaussian_normalized_noexp(y_pred, y_true, show_parts=True):
    mean_true = tf.math.divide(y_true[:, :, :, 0], 2**16)
    mean_pred = tf.math.divide(y_pred[:, :, :, 0], 2**16)
    scale_pred = K.pow(tf.math.divide(y_pred[:, :, :, 1], 2**16), 2)
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


def mae_on_first_channel(y_true, y_pred):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    loss = K.abs(mean_true - mean_pred)
    return loss


# both y_true and y_pred have shape of batch_size * rows * cols * 2
# in y_true , the first channel is gt mean, second is binary mask where 0 indicates phase wrapping
# in y_pred , the first channel is predicted mean, the second is predicted scale corresponds to
# per-pixel laplace distribution
def laplacian_loss_with_mask(y_true, y_pred):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    scale_pred = y_pred[:, :, :, 1]
    mask = y_true[:, :, :, 1]
    loss = K.tf.divide(K.abs(mean_true - mean_pred), scale_pred + 1e-7) + K.log(
        scale_pred + 1e-7
    )
    loss = K.tf.multiply(loss, mask)
    return loss


# simple mae loss to monitor spatial behavior
def mae_with_mask(y_true, y_pred):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    loss = K.tf.multiply(K.abs(mean_true - mean_pred), mask)
    return loss


def my_accuracy(y_true, y_pred):
    return K.equal(K.round(y_true), K.round(y_pred))


def gradient_loss(y_true, y_pred):
    y_true_gradient_horizontal = K.abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :])
    y_true_gradient_vertical = K.abs(y_true[:, 1:, :, :] - y_true[:, :-1, :, :])
    y_pred_gradient_horizontal = K.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    y_pred_gradient_vertical = K.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])

    loss_1 = K.abs(y_true_gradient_horizontal - y_pred_gradient_horizontal)
    loss_2 = K.abs(y_true_gradient_vertical - y_pred_gradient_vertical)
    loss_1 = K.tf.reduce_mean(K.tf.reduce_mean(K.tf.reduce_mean(loss_1, 3), 2), 1)
    loss_2 = K.tf.reduce_mean(K.tf.reduce_mean(K.tf.reduce_mean(loss_2, 3), 2), 1)
    return loss_1 + loss_2
