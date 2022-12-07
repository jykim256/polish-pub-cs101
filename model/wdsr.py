import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, ReLU, SpatialDropout2D
from tensorflow.python.keras.models import Model

from model.common import decenter, denormalize, normalize, pixel_shuffle


def wdsr_b_uq_norelu(
    scale,
    num_filters=32,
    num_res_blocks=8,
    res_block_expansion=6,
    res_block_scaling=None,
    nchan=1,
    output_chan=2,
):

    x_in = Input(shape=(None, None, nchan))
    x = Lambda(normalize)(x_in)

    # main branch
    #    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    m = conv2d_weightnorm(num_filters, nchan, padding="same")(x)
    for i in range(num_res_blocks):
        m = res_block_b(
            m,
            num_filters,
            res_block_expansion,
            kernel_size=3,
            scaling=res_block_scaling,
        )
    m = conv2d_weightnorm(
        output_chan * nchan * scale**2,
        3,
        padding="same",
        name=f"conv2d_main_scale_{scale}",
    )(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(
        output_chan * nchan * scale**2,
        5,
        padding="same",
        name=f"conv2d_skip_scale_{scale}",
    )(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    # x = Lambda(denormalize)(x)
    # x = Lambda(decenter)(x)
    # x = ReLU()(x)

    return Model(x_in, x, name="wdsr_b_uq_norelu")


def wdsr_b_uq_norelu_dropout(
    scale,
    num_filters=32,
    num_res_blocks=8,
    res_block_expansion=6,
    res_block_scaling=None,
    nchan=1,
    output_chan=2,
):

    x_in = Input(shape=(None, None, nchan))
    x = Lambda(normalize)(x_in)

    # main branch
    #    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    m = conv2d_weightnorm_dropout(num_filters, nchan, padding="same")(x)
    for i in range(num_res_blocks):
        m = res_block_b_dropout(
            m,
            num_filters,
            res_block_expansion,
            kernel_size=3,
            scaling=res_block_scaling,
        )
    m = conv2d_weightnorm_dropout(
        output_chan * nchan * scale**2,
        3,
        padding="same",
        name=f"conv2d_main_scale_{scale}",
    )(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm_dropout(
        output_chan * nchan * scale**2,
        5,
        padding="same",
        name=f"conv2d_skip_scale_{scale}",
    )(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    # x = Lambda(denormalize)(x)
    # x = Lambda(decenter)(x)
    # x = ReLU()(x)

    return Model(x_in, x, name="wdsr_b_uq_norelu_dropout")


def wdsr_b_uq(
    scale,
    num_filters=32,
    num_res_blocks=8,
    res_block_expansion=6,
    res_block_scaling=None,
    nchan=1,
    output_chan=2,
):

    x_in = Input(shape=(None, None, nchan))
    x = Lambda(normalize)(x_in)

    # main branch
    #    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    m = conv2d_weightnorm(num_filters, nchan, padding="same")(x)
    for i in range(num_res_blocks):
        m = res_block_b(
            m,
            num_filters,
            res_block_expansion,
            kernel_size=3,
            scaling=res_block_scaling,
        )
    m = conv2d_weightnorm(
        output_chan * nchan * scale**2,
        3,
        padding="same",
        name=f"conv2d_main_scale_{scale}",
    )(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(
        output_chan * nchan * scale**2,
        5,
        padding="same",
        name=f"conv2d_skip_scale_{scale}",
    )(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    # x = Lambda(denormalize)(x)
    x = Lambda(decenter)(x)
    x = ReLU()(x)

    return Model(x_in, x, name="wdsr_b_uq")


def wdsr_b_uq_norelu_mc(
    scale,
    num_filters=32,
    num_res_blocks=8,
    res_block_expansion=6,
    res_block_scaling=None,
    nchan=1,
    output_chan=2,
    dropout_rate=0
):
    print('DROP OUT RATE: ', dropout_rate)

    x_in = Input(shape=(None, None, nchan))
    x = Lambda(normalize)(x_in)

    # main branch
    #    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    m = conv2d_weightnorm(num_filters, nchan, padding="same")(x)
    for i in range(num_res_blocks):
        m = res_block_b_dropout(
            m,
            num_filters,
            res_block_expansion,
            kernel_size=3,
            scaling=res_block_scaling,
            dropout_rate=dropout_rate,
        )
    m = dropout_mc_wrapper(m, rate=dropout_rate)
    m = conv2d_weightnorm(
        output_chan * nchan * scale**2,
        3,
        padding="same",
        name=f"conv2d_main_scale_{scale}",
    )(m)
    m = dropout_mc_wrapper(m, rate=dropout_rate)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(
        output_chan * nchan * scale**2,
        5,
        padding="same",
        name=f"conv2d_skip_scale_{scale}",
    )(x)
    s = dropout_mc_wrapper(s, rate=dropout_rate)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    # x = Lambda(denormalize)(x)
    # x = Lambda(decenter)(x)
    # x = ReLU()(x)

    return Model(x_in, x, name="wdsr_b_uq_norelu_mc")


def dropout_mc_wrapper(x, rate=0.1):
    # print('Dropout being used!')
    # return tf.nn.dropout(x, rate)
    return SpatialDropout2D(rate)(x, training=True)


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weightnorm(
        num_filters * expansion, 1, padding="same", activation="relu"
    )(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding="same")(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding="same")(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm(filters, kernel_size, padding="same", activation=None, **kwargs):
    return tfa.layers.WeightNormalization(
        Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs),
        data_init=False,
    )


def res_block_b_dropout(
    x_in, num_filters, expansion, kernel_size, scaling, dropout_rate=0.1
):
    linear = 0.8
    x = conv2d_weightnorm(
        num_filters * expansion, 1, padding="same", activation="relu"
    )(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding="same")(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding="same")(x)
    x = dropout_mc_wrapper(x, rate=dropout_rate)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm_dropout(
    filters, kernel_size, padding="same", activation=None, dropout_rate=0.1, **kwargs
):
    return (tfa.layers.WeightNormalization(
        Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs),
        data_init=False,
    ))
