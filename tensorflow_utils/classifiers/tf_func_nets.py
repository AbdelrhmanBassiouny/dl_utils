import tensorflow as tf
from tensorflow_utils.classifiers.tf_func_layers import DLUtils


def mini_alex_net(shape=(32, 32, 3), num_classes=10, conv_params=((32, 3), (32, 3),
                                                                  (64, 3), (64, 3),
                                                                  (128, 3), (128, 3)),
                  affine_size=None, use_avg_pool=False,
                  l1=None, l2=None, drop_prop=None, activate_first=False):

    reg = {'l1': l1, 'l2': l2, 'drop_prop': drop_prop}

    dlu = DLUtils(**reg, activate_first=activate_first)

    input_ = tf.keras.layers.Input(shape=shape)

    x = tf.keras.layers.BatchNormalization()(input_)

    for i in tf.range(0, len(conv_params) - 1):
        x = dlu.conv_normalize_drop(x, *conv_params[i])
        if (i+1) % 2 == 0:
            x = tf.keras.layers.MaxPool2D(2, strides=2, padding='valid')(x)

    x = dlu.conv_normalize_drop(x, *conv_params[-1])

    if not use_avg_pool:
        x = tf.keras.layers.Flatten()(x)
    else:
        x = tf.keras.layers.GlobalAvgPool2D()(x)

    if affine_size:
        for i in tf.range(0, len(affine_size)):
            x = dlu.dense_normalize_drop(x, affine_size[i])

    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=[input_], outputs=[x])


def mini_inception(shape=(32, 32, 3), num_classes=10, params=(64, 32, 16, 32, 16, 64),
                   naive=False, l1=None, l2=None, drop_prop=None, activate_first=False, n_modules=3):

    reg = {'l1': l1, 'l2': l2, 'drop_prop': drop_prop}

    dlu = DLUtils(**reg, activate_first=activate_first)

    input_ = tf.keras.layers.Input(shape=shape)

    x = tf.keras.layers.BatchNormalization()(input_)

    inception = dlu.naive_inception_module if naive else dlu.inception_module

    for _ in tf.range(0, n_modules):
        x = inception(x, *params)
        x = inception(x, *params)
        x = tf.keras.layers.MaxPool2D(2, strides=2, padding='valid')(x)
        params = tuple([p * 2 for p in params])

    x = inception(x, *params)
    x = inception(x, *params)
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    return tf.keras.Model(inputs=[input_], outputs=[x])


def res_net(shape=(32, 32, 3), num_classes=10, first_block_filters=32, filter_sz=3,
            repeats=2, reduce_every=2,
            l1=None, l2=None, drop_prop=None, naive=True, activate_first=False):

    reg = {'l1': l1, 'l2': l2, 'drop_prop': drop_prop}
    dlu = DLUtils(**reg, activate_first=activate_first)

    in_ch = first_block_filters
    input_ = tf.keras.layers.Input(shape=shape)

    x = tf.keras.layers.BatchNormalization()(input_)

    res = dlu.naive_res_block if naive else dlu.res_block

    x = res(x, in_ch, filter_sz, project=True)
    for _ in range(0, reduce_every - 1):
        x = res(x, in_ch, filter_sz)

    for _ in tf.range(0, repeats):
        in_ch *= 2
        x = res(x, in_ch, filter_sz, reduce=True)
        for _ in tf.range(0, reduce_every - 1):
            x = res(x, in_ch, filter_sz)

    x = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=[input_], outputs=[x])


def dense_net(shape=(32, 32, 3), num_classes=10, k=12, dense_layers=(6, 12, 24), compactness=0.5, pool='avg',
              l1=None, l2=None, drop_prop=None, activate_first=False):

    reg = {'l1': l1, 'l2': l2, 'drop_prop': drop_prop}
    dlu = DLUtils(**reg, activate_first=activate_first)

    input_ = tf.keras.layers.Input(shape=shape)
    n_channels = shape[-1]

    x, n_channels = dlu.dense_block(input_, n_channels, dense_layers[0], k=k)
    for i in tf.range(1, len(dense_layers)):
        x, n_channels = dlu.transition_block(x, n_channels, compactness=compactness, pool=pool)
        x, n_channels = dlu.dense_block(x, n_channels, dense_layers[i], k=k)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=[input_], outputs=[x])
