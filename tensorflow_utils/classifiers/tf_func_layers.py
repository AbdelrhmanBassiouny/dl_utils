from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Conv2D, Dense, SpatialDropout2D,\
    BatchNormalization, MaxPool2D, AvgPool2D, Concatenate, Add
import tensorflow as tf


class DLUtils:
    def __init__(self, drop_prop=None, l1=None, l2=None, activation='relu', activate_first=False):
        self.l1, self.l2, self.drop_prop = l1, l2, drop_prop
        self.activation = activation
        self.activate_first = activate_first

    def normalize_activate_drop(self, x):

        if self.activate_first:
            x = Activation(self.activation)(x)
            x = BatchNormalization()(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)

        if self.drop_prop:
            x = SpatialDropout2D(rate=self.drop_prop)(x)
        return x

    def conv_normalize_drop(self, x, filters, size, pad='same', stride=1):

        k, f, s, p = filters, size, stride, pad
        reg = regularizers.L1L2(l1=self.l1, l2=self.l2)
        x = Conv2D(k, f, s, p, kernel_initializer='he_normal', kernel_regularizer=reg, use_bias=False)(x)
        x = self.normalize_activate_drop(x)
        return x

    def dense_normalize_drop(self, x, units):
        reg = regularizers.L1L2(l1=self.l1, l2=self.l2)
        x = Dense(units, kernel_initializer='he_normal', kernel_regularizer=reg, use_bias=False)(x)
        x = self.normalize_activate_drop(x)
        return x

    def _template_inception_module(self, x, filters1x1, filters3x3, filters5x5,
                                   filters3x3r=None, filters5x5r=None, filters_pool_proj=None, naive=True):

        if not naive:
            conv3x3r = self.conv_normalize_drop(x, filters3x3r, 1)
            conv5x5r = self.conv_normalize_drop(x, filters5x5r, 1)
        else:
            conv3x3r, conv5x5r = x, x

        conv1x1 = self.conv_normalize_drop(x, filters1x1, 1)
        conv3x3 = self.conv_normalize_drop(conv3x3r, filters3x3, 3)
        conv5x5 = self.conv_normalize_drop(conv5x5r, filters5x5, 5)
        pool = MaxPool2D(3, padding='same', strides=1)(x)

        if not naive:
            pool = self.conv_normalize_drop(pool, filters_pool_proj, 1)

        return Concatenate()((conv1x1, conv3x3, conv5x5, pool))

    def naive_inception_module(self, x, filters1x1, filters3x3, filters5x5):
        return self._template_inception_module(x, filters1x1, filters3x3, filters5x5)

    def inception_module(self, x, filters1x1, filters3x3, filters5x5, filters3x3r, filters5x5r, filters_pool_proj):
        return self._template_inception_module(x, filters1x1, filters3x3, filters5x5, filters3x3r, filters5x5r,
                                               filters_pool_proj, naive=False)

    def _template_res_block(self, x, in_channels, filter_sz, conv1_fz=1, out_channels=None,
                            reduce=False, project=False):
        """
        Only implements options (B) and (C) of the resnet paper.
        Projector Conv is a (1,1) conv that adjusts H, W, and no. of channels to be same as output.
        It's only used when those dimensions are changed from input to output.
        """

        # Adjust first conv layer parameters depending on whether it will reduce spatial dims or not.
        s = 2 if reduce else 1
        proj_s = 2 if reduce else 1

        out = self.conv_normalize_drop(x, in_channels, conv1_fz, stride=s)
        out = self.conv_normalize_drop(out, in_channels, filter_sz)

        proj_ch = in_channels
        if out_channels:
            out = self.conv_normalize_drop(out, out_channels, 1)
            proj_ch = out_channels

        if reduce or project:
            x = self.conv_normalize_drop(x, proj_ch, 1, 'valid', proj_s)

        return Add()([out, x])

    def naive_res_block(self, x, in_channels, filter_sz, reduce=False, project=False):
        return self._template_res_block(x, in_channels, filter_sz, conv1_fz=filter_sz, reduce=reduce, project=project)

    def res_block(self, x, in_channels, filter_sz, out_channels, reduce=False, project=False):
        return self._template_res_block(x, in_channels, filter_sz,
                                        out_channels=out_channels, reduce=reduce, project=project)

    def conv_block(self, x, filters, bottle_neck_filters=None, pad='same', stride=1):
        if bottle_neck_filters:
            x = self.conv_normalize_drop(x, bottle_neck_filters, 1, pad=pad, stride=stride)
        x = self.conv_normalize_drop(x, filters, 3)
        return x

    def dense_block(self, x, in_channels, num_layers, k=12):
        inputs = x
        for _ in tf.range(0, num_layers):
            x = self.conv_block(inputs, k, 4*k)
            inputs = Concatenate()([inputs, x])
        out_channels = in_channels + k * num_layers
        return inputs, out_channels

    def transition_block(self, x, in_channels, compactness=0.5, pool='avg'):
        out_channels = in_channels * compactness
        x = self.conv_normalize_drop(x, out_channels, 1)
        x = AvgPool2D(strides=2)(x) if pool == 'avg' else MaxPool2D(strides=2)(x)
        return x, out_channels
