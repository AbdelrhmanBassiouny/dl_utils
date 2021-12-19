import tensorflow as tf
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import Activation, Conv2D, Dense, SpatialDropout2D,\
    BatchNormalization, MaxPool2D, AvgPool2D, Concatenate, Add


class _NormalizeActivateDrop(tf.keras.layers.Layer):
    def __init__(self, l1=0.0, l2=0.0, drop_prop=None,
                 activation='relu', activate_first=False, **kwargs):
        super(_NormalizeActivateDrop, self).__init__(**kwargs)

        self.l1, self.l2, self.drop_prop = l1, l2, drop_prop
        self.activation, self.activate_first = activation, activate_first

        self.initializer = initializers.variance_scaling(scale=2.0)
        self.reg = regularizers.l1_l2(l1=self.l1, l2=self.l2)

        self.activate = Activation(self.activation)

        self.bn = BatchNormalization()

        if self.drop_prop:
            self.dropout = SpatialDropout2D(rate=self.drop_prop)

    def get_config(self):
        config = super(_NormalizeActivateDrop, self).get_config()
        config.update({
            'l1': self.l1,
            'l2': self.l2,
            'drop_prop': self.drop_prop,
            'activation': self.activation,
            'activate_first': self.activate_first
        })
        return config

    def call(self, x, training=None):

        if self.activate_first:
            out = self.activate(x)
            out = self.bn(out, training=training)
        else:
            out = self.bn(x, training=training)
            out = self.activate(out)

        if self.drop_prop:
            out = self.dropout(out, training=training)
        return out


class ConvNormalizeDrop(_NormalizeActivateDrop):
    def __init__(self, filters, size, pad='same', stride=1, **kwargs):
        super(ConvNormalizeDrop, self).__init__(**kwargs)

        self.K, self.F, self.p, self.s = filters, size, pad, stride

        self.conv = Conv2D(self.K, (self.F, self.F), padding=self.p, strides=self.s,
                           kernel_initializer=self.initializer,
                           kernel_regularizer=self.reg)

    def call(self, x, training=None):
        out = self.conv(x)
        out = super(ConvNormalizeDrop, self).call(out, training=training)
        return out

    def get_config(self):
        config = super(ConvNormalizeDrop, self).get_config()
        config.update({'filters': self.K, 'size': self.F, 'pad': self.p, 'stride': self.s})
        return config


class DenseNormalizeDrop(_NormalizeActivateDrop):
    def __init__(self, units, **kwargs):
        super(DenseNormalizeDrop, self).__init__(**kwargs)

        self.units = units
        self.dense = Dense(self.units, kernel_initializer=self.initializer,
                           kernel_regularizer=self.reg)

    def call(self, x, training=None):
        out = self.dense(x)
        out = super(DenseNormalizeDrop, self).call(out, training=training)
        return out

    def get_config(self):
        config = super(DenseNormalizeDrop, self).get_config()
        config.update({'units': self.units})
        return config


class _TemplateInceptionModule(tf.keras.layers.Layer):
    def __init__(self, filters1x1, filters3x3, filters5x5,
                 filters3x3r=None, filters5x5r=None, filters_pool_proj=None,
                 naive=True, **kwargs):
        super(_TemplateInceptionModule, self).__init__()

        self.f1x1, self.f3x3, self.f5x5 = filters1x1, filters3x3, filters5x5
        self.f3x3r, self.f5x5r, self.f_pool = filters3x3r, filters5x5r, filters_pool_proj
        self.naive = naive

        if not self.naive:
            self.conv3x3r = ConvNormalizeDrop(self.f3x3r, 1, **kwargs)
            self.conv5x5r = ConvNormalizeDrop(self.f5x5r, 1, **kwargs)
            self.pool_proj = ConvNormalizeDrop(self.f_pool, 1, **kwargs)

        self.conv1x1 = ConvNormalizeDrop(self.f1x1, 1, **kwargs)
        self.conv3x3 = ConvNormalizeDrop(self.f3x3, 3, **kwargs)
        self.conv5x5 = ConvNormalizeDrop(self.f5x5, 5, **kwargs)
        self.pool = MaxPool2D((3, 3), padding='same', strides=(1, 1))
        self.concat = Concatenate()

    def call(self, x, training=None):

        out1x1 = self.conv1x1(x, training=training)

        if not self.naive:
            out3x3r = self.conv3x3r(x, training=training)
            out5x5r = self.conv5x5r(x, training=training)
        else:
            out3x3r, out5x5r = x, x
        out3x3 = self.conv3x3(out3x3r, training=training)
        out5x5 = self.conv5x5(out5x5r, training=training)

        out_pool = self.pool(x)
        if not self.naive:
            out_pool = self.pool_proj(out_pool, training=training)

        return self.concat((out1x1, out3x3, out5x5, out_pool))

    def get_config(self):
        config = super(_TemplateInceptionModule, self).get_config()
        config.update({self.f1x1: 'filters1x1', self.f3x3: 'filters3x3', self.f5x5: 'filters5x5',
                       self.f3x3r: 'filters5x5r', self.f5x5r: 'filters_pool_proj',
                       self.f_pool: 'filters3x3r', self.naive: 'naive'})
        return config


class NaiveInceptionModule(_TemplateInceptionModule):
    def __init__(self, filters1x1, filters3x3, filters5x5, **kwargs):
        super(NaiveInceptionModule, self).__init__(filters1x1, filters3x3, filters5x5, **kwargs)


class InceptionModule(_TemplateInceptionModule):
    def __init__(self, filters1x1, filters3x3, filters5x5,
                 filters3x3r, filters5x5r, filters_pool_proj, **kwargs):
        super(InceptionModule, self).__init__(filters1x1, filters3x3, filters5x5,
                                              filters3x3r, filters5x5r,
                                              filters_pool_proj, naive=False, **kwargs)


class _TemplateResBlock(tf.keras.layers.Layer):
    """
    Only implements options (B) and (C) of the resnet paper.
    Projector Conv is a (1,1) conv that adjusts H, W, and no. of channels to be same as output.
    It's only used when those dimensions are changed from input to output.
    """
    def __init__(self, in_channels, filter_sz,  conv1_fz=1, out_channels=None, reduce=False, project=False, **kwargs):
        super(_TemplateResBlock, self).__init__()

        # Record inputs for get_config method.
        self.in_ch, self.F, self.conv1_fz = in_channels, filter_sz, conv1_fz
        self.reduce, self.project = reduce, project
        self.out_ch = out_channels

        # Adjust first conv layer parameters depending on whether it will reduce spatial dims or not.
        s = 2 if self.reduce else 1

        self.conv1 = ConvNormalizeDrop(self.in_ch, self.conv1_fz, stride=s, **kwargs)
        self.conv2 = ConvNormalizeDrop(self.in_ch, self.F, **kwargs)

        proj_ch = self.in_ch
        if self.out_ch:
            self.conv3 = ConvNormalizeDrop(self.out_ch, 1, **kwargs)
            proj_ch = self.out_ch

        if self.reduce or self.project:
            self.projector_conv = ConvNormalizeDrop(proj_ch, 1, pad='valid', stride=s, **kwargs)

        self.add = tf.keras.layers.Add()

    def call(self, x, training=None):
        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)
        if self.reduce or self.project:
            x = self.projector_conv(x, training=training)
        if self.out_ch:
            out = self.conv3(out, training=training)
        return self.add([out, x])

    def get_config(self):
        config = super(_TemplateResBlock, self).get_config()
        config.update({
            'in_channels': self.in_ch,
            'filter_sz': self.F,
            'conv1_fz': self.conv1_fz,
            'out_channels': self.out_ch,
            'reduce': self.reduce,
            'project': self.project
        })
        return config


class NaiveResBlock(_TemplateResBlock):
    def __init__(self, in_channels, filter_sz, reduce=False, project=False, **kwargs):
        super(NaiveResBlock, self).__init__(in_channels, filter_sz, conv1_fz=filter_sz,
                                            reduce=reduce, project=project, **kwargs)


class ResBlock(_TemplateResBlock):
    def __init__(self, in_channels, filter_sz, out_channels, reduce=False, project=False, **kwargs):
        super(ResBlock, self).__init__(in_channels, filter_sz, out_channels=out_channels,
                                       reduce=reduce, project=project, **kwargs)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, bottle_neck_filters=None, pad='same', stride=1, **kwargs):
        super(ConvBlock, self).__init__()
        self.filters, self.bottle_neck_filters = filters, bottle_neck_filters
        self.p, self.s = pad, stride
        if self.bottle_neck_filters:
            self.bottle_neck_conv = ConvNormalizeDrop(self.bottle_neck_filters, 1,
                                                      pad=self.p, stride=self.s, **kwargs)
        self.conv = ConvNormalizeDrop(self.filters, 3, **kwargs)

    def call(self, x, training=None, mask=None):
        if self.bottle_neck_filters:
            x = self.bottle_neck_conv(x, training=training)
        return self.conv(x, training=training)

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({'filters': self.filters, 'bottle_neck_filters': self.bottle_neck_filters,
                       'pad': self.p, 'stride': self.s})
        return config


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_layers, k=12, **kwargs):
        super(DenseBlock, self).__init__()
        self.in_ch = in_channels
        self.num_l = num_layers
        self.k = k

        self.conv_blocks, self.concat_layers = [], []
        for _ in tf.range(0, num_layers):
            self.conv_blocks.append(ConvBlock(k, 4*k, **kwargs))
            self.concat_layers.append(Concatenate())
        self.out_ch = self.in_ch + self.k * self.num_l

    def call(self, x, training=None, mask=None):
        inputs = x
        for layer, concat in zip(self.conv_blocks, self.concat_layers):
            x = layer(inputs, training=training)
            inputs = concat([inputs, x])
        return inputs

    def get_config(self):
        config = super(DenseBlock, self).get_config()
        config.update({'in_channels': self.in_ch, 'num_layers': self.num_l, 'k': self.k})
        return config



class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, compactness=0.5, pool='avg', **kwargs):
        super(TransitionBlock, self).__init__()
        self.in_ch = in_channels
        self.c = compactness
        self.pool_t = pool

        self.out_ch = self.in_ch * self.c
        self.conv = ConvNormalizeDrop(self.out_ch, 1, **kwargs)
        self.pool = AvgPool2D(strides=2) if pool == 'avg' else MaxPool2D(strides=2)

    def call(self, x, training=None, mask=None):
        x = self.conv(x, training=training)
        return self.pool(x)

    def get_config(self):
        config = super(TransitionBlock, self).get_config()
        config.update({'in_channels': self.in_ch, 'compactness': self.c, 'pool': self.pool_t})
        return config