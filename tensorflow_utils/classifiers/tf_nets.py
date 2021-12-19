import tensorflow as tf
from tf_layers import DenseNormalizeDrop, ConvNormalizeDrop,\
    NaiveInceptionModule, InceptionModule, NaiveResBlock, ResBlock,\
    DenseBlock, TransitionBlock
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Flatten, ZeroPadding2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomCrop, RandomFlip, Resizing, Normalization, RandomTranslation


class KerasModel(tf.keras.Model):
    def __init__(self):
        super(KerasModel, self).__init__()

    def get_functional_model(self, input_shape, preprocessor=None):
        input_ = tf.keras.layers.Input(shape=input_shape[1:])
        x = input_
        if preprocessor:
            x = preprocessor(input_)
        scores = self.call(x)
        return tf.keras.Model(inputs=[input_], outputs=[scores])


class MiniAlexNet(KerasModel):
    def __init__(self, num_classes=10, conv_params=((32, 3), (32, 3),
                                                    (64, 3), (64, 3),
                                                    (128, 3), (128, 3)),
                 affine_size=None, use_avg_pool=False,
                 l1=None, l2=None, drop_prop=None, activate_first=False):
        super(MiniAlexNet, self).__init__()

        self.use_avg_pool = use_avg_pool
        reg = {'l1': l1, 'l2': l2, 'drop_prop': drop_prop, 'activate_first': activate_first}

        self.conv_layers = []
        self.conv_layers.append(ConvNormalizeDrop(*conv_params[0], **reg))
        self.conv_layers.append(ConvNormalizeDrop(*conv_params[1], **reg))
        for i in tf.range(2, len(conv_params)):
            if i % 2 == 0:
                self.conv_layers.append(tf.keras.layers.MaxPool2D(strides=2))
            self.conv_layers.append(ConvNormalizeDrop(*conv_params[i], **reg))

        if self.use_avg_pool:
            self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        else:
            self.flatten = tf.keras.layers.Flatten()

        self.affine_layers = []
        if affine_size:
            for i in range(len(affine_size)):
                self.affine_layers.append(DenseNormalizeDrop(affine_size[i], **reg))

        self.scores = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=None, mask=None):

        for layer in self.conv_layers:
            x = layer(x, training=training)

        if self.use_avg_pool:
            x = self.avg_pool(x)
        else:
            x = self.flatten(x)

        for layer in self.affine_layers:
            x = layer(x, training=training)

        return self.scores(x)


class MiniInception(KerasModel):
    def __init__(self, num_classes=10, params=(64, 32, 16, 32, 16, 64),
                 naive=False, l1=None, l2=None, drop_prop=None, activate_first=False, n_modules=3):
        super(MiniInception, self).__init__()

        self.naive = naive
        reg = {'l1': l1, 'l2': l2, 'drop_prop': drop_prop, 'activate_first': activate_first}
        self.nc = num_classes

        inception = NaiveInceptionModule if naive else InceptionModule

        self.inception_modules = []
        for _ in tf.range(0, n_modules):
            self.inception_modules.append(inception(*params, **reg))
            self.inception_modules.append(inception(*params, **reg))
            self.inception_modules.append(tf.keras.layers.MaxPool2D(2, strides=2, padding='valid'))
            params = tuple([p * 2 for p in params])

        self.inception_modules.append(inception(*params, **reg))
        self.inception_modules.append(inception(*params, **reg))
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.fc = tf.keras.layers.Dense(self.nc, activation='softmax', name='output')

    def call(self, x, training=None, mask=None):
        out = x
        for layer in self.inception_modules:
            out = layer(out, training=training)
        out = self.avg_pool(out)
        return self.fc(out)


class ResNet(KerasModel):
    def __init__(self, num_classes=10, first_block_filters=32, filter_sz=3,
                 repeats=2, reduce_every=2,
                 l1=None, l2=None, drop_prop=None, naive=True, activate_first=False):
        super(ResNet, self).__init__()
        self.nc = num_classes
        self.repeats, self.reduce_every = repeats, reduce_every
        self.in_ch, self.F = first_block_filters, filter_sz

        reg = {'l1': l1, 'l2': l2, 'drop_prop': drop_prop, 'activate_first': activate_first}

        in_ch = first_block_filters

        res = NaiveResBlock if naive else ResBlock

        self.res_blocks = []
        self.res_blocks.append(res(in_ch, filter_sz, project=True, **reg))
        for _ in range(0, reduce_every - 1):
            self.res_blocks.append(res(in_ch, filter_sz, **reg))

        for _ in tf.range(0, repeats):
            in_ch *= 2
            self.res_blocks.append(res(in_ch, filter_sz, reduce=True, **reg))
            for _ in tf.range(0, reduce_every - 1):
                self.res_blocks.append(res(in_ch, filter_sz, **reg))

        self.avg_pool = tf.keras.layers.GlobalAvgPool2D()
        # This initializer should be changed, since this is for 'relu' activation layers.
        self.scores = tf.keras.layers.Dense(self.nc, activation='softmax')

    def call(self, x, training=None, mask=None):
        out = x
        for layer in self.res_blocks:
            out = layer(out, training=training)
        out = self.avg_pool(out)
        return self.scores(out)


class DenseNet(KerasModel):
    def __init__(self, shape=(32, 32, 3),  num_classes=10, k=12, dense_layers=(6, 12, 24), compactness=0.5, pool='avg',
                 l1=None, l2=None, drop_prop=None, activate_first=False):
        super(DenseNet, self).__init__()

        reg = {'l1': l1, 'l2': l2, 'drop_prop': drop_prop, 'activate_first': activate_first}
        n_channels = shape[-1]
        self.dense_layers = [DenseBlock(n_channels, dense_layers[0], k=k, **reg)]
        n_channels = self.layers[-1].out_ch
        for i in tf.range(1, len(dense_layers)):
            self.dense_layers.append(TransitionBlock(n_channels, compactness=compactness, pool=pool, **reg))
            n_channels = self.layers[-1].out_ch
            self.dense_layers.append(DenseBlock(n_channels, dense_layers[i], k=k, **reg))
            n_channels = self.layers[-1].out_ch

        self.avg_pool = tf.keras.layers.GlobalAvgPool2D()
        self.scores = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=None, mask=None):
        for layer in self.dense_layers:
            x = layer(x, training=training)
        x = self.avg_pool(x)
        return self.scores(x)


class PreTrainedModel(KerasModel):
    def __init__(self, model='densenet121', shape=(32, 32, 3), num_classes=10, drop_prop=None, fine_tune=None):
        super(PreTrainedModel, self).__init__()
        self.drop = drop_prop
        self.shape = shape
        if model == 'densenet121':
            self.base_model = DenseNet121(include_top=False, input_shape=shape, classes=num_classes)

        if fine_tune:
            self.base_model.trainable = True
            # Train layers starting from layer[fine_tune] onwards.
            for layer in self.base_model.layers[:fine_tune]:
                layer.trainable = False
        else:
            self.base_model.trainable = False

        if self.drop:
            self.dropout = tf.keras.layers.Dropout(rate=drop_prop)

        self.avg_pool = tf.keras.layers.GlobalAvgPool2D()

        self.prediction_layer = Dense(num_classes, activation='softmax')

    def call(self, x, training=None, mask=None):
        x = self.base_model(x, training=training)
        x = self.avg_pool(x)
        if self.drop:
            x = self.dropout(x, training=training)
        return self.prediction_layer(x)


class ResizedCrop(tf.keras.layers.Layer):
    def __init__(self, crop_to=(24, 24), resize_to=(32, 32)):
        super(ResizedCrop, self).__init__()

        self.crop_to = crop_to
        self.resize_to = resize_to

        if self.crop_to is not None:
            self.cropper = RandomCrop(*self.crop_to)
        if self.resize_to is not None:
            self.resizer = Resizing(*self.resize_to)

    def call(self, x, training=None, mask=None):
        if self.crop_to:
            x = self.cropper(x, training=training)
        if self.resize_to:
            x = self.resizer(x, training=training)
        return x


class Augmenter(tf.keras.layers.Layer):
    def __init__(self, resized_crop=(24, 24), translate_by=(0.25, 0.25), fill_mode='reflect', horizontal_flip=True):
        super(Augmenter, self).__init__()

        self.horizontal_flip = horizontal_flip
        self.translate_by = translate_by
        self.resized_crop = resized_crop

        if self.horizontal_flip:
            self.h_flipper = RandomFlip("horizontal")

        if self.translate_by is not None:
            self.translator = RandomTranslation(*translate_by, fill_mode=fill_mode)

        if self.resized_crop is not None:
            self.resized_cropper = ResizedCrop(self.resized_crop)

    def call(self, x, training=None, mask=None):
        if self.horizontal_flip:
            x = self.h_flipper(x, training=training)
        if self.translate_by:
            x = self.translator(x, training=training)
        if self.resized_crop:
            x = self.resized_cropper(x, training=training)
        return x


class ResizeAndNormalize(tf.keras.layers.Layer):
    def __init__(self, resize_to=None, interp='bilinear', normalize=True, rescale=True):
        super(ResizeAndNormalize, self).__init__()

        self.resize_to = resize_to
        self.normalize = normalize
        self.rescale = rescale

        if self.rescale:
            self.rescaler = Rescaling(1.0/255.0)

        if self.resize_to:
            self.resizer = Resizing(*resize_to, interpolation=interp)

        if self.normalize:
            self.normalizer = Normalization()

    def adapt_normalizer(self, data, adapt_to_resized=True, reset_state=False):
        if adapt_to_resized:
            self.normalize = False
            data = self.call(data)
            self.normalize = True
        self.normalizer.adapt(data, reset_state=reset_state)

    def call(self, x):
        if self.rescale:
            x = self.rescaler(x)
        if self.resize_to:
            x = self.resizer(x)
        if self.normalize:
            x = self.normalizer(x)
        return x
