import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1DTranspose, Conv1D, BatchNormalization, Dense, Reshape
from tensorflow.keras.models import Model
class WaveGANGenerator(Model):
    def __init__(self, slice_len=16384, nch=1, kernel_len=25, dim=64,
                 use_batchnorm=False, upsample='zeros'):
        super(WaveGANGenerator, self).__init__()
        assert slice_len in [16384, 32768, 65536]
        self.slice_len = slice_len
        self.nch = nch
        self.kernel_len = kernel_len
        self.dim = dim
        self.use_batchnorm = use_batchnorm
        self.upsample = upsample

        dim_mul = 16 if slice_len == 16384 else 32
        self.fc = layers.Dense(16 * dim * dim_mul)
        self.reshape = layers.Reshape((16, dim * dim_mul))
        self.bn_layers = []

        def bn_layer():
            return layers.BatchNormalization() if use_batchnorm else tf.identity

        self.upconvs = []
        for _ in range(4):
            dim_mul //= 2
            self.upconvs.append((dim * dim_mul, bn_layer()))
        if slice_len == 16384:
            self.final_layers = [(nch, None)]
        elif slice_len == 32768:
            self.final_layers = [
                (dim, bn_layer()),
                (nch, None)
            ]
        elif slice_len == 65536:
            self.final_layers = [
                (dim, bn_layer()),
                (nch, None)
            ]

    def conv1d_transpose(self, inputs, filters, kernel_size, stride):
        # Use Conv2DTranspose and reshape to simulate 1D transposed convolution
        x = tf.expand_dims(inputs, axis=1)  # (B, 1, T, C)
        x = layers.Conv2DTranspose(filters=filters,
                                   kernel_size=(1, kernel_size),
                                   strides=(1, stride),
                                   padding='same')(x)
        return x[:, 0]  # Remove dummy spatial dimension

    def call(self, z, training=False):
        dim_mul = 16 if self.slice_len == 16384 else 32
        x = self.fc(z)
        x = self.reshape(x)
        if self.use_batchnorm:
            x = layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        dim_mul //= 2

        # Main upsampling blocks
        for filters, bn in self.upconvs:
            x = self.conv1d_transpose(x, filters, self.kernel_len, 4)
            if self.use_batchnorm:
                x = bn(x, training=training)
            x = tf.nn.relu(x)

        # Final layer(s)
        if self.slice_len == 16384:
            filters, _ = self.final_layers[0]
            x = self.conv1d_transpose(x, filters, self.kernel_len, 4)
        elif self.slice_len == 32768:
            for i, (filters, bn) in enumerate(self.final_layers):
                stride = 4 if i == 0 else 2
                x = self.conv1d_transpose(x, filters, self.kernel_len, stride)
                if bn is not None:
                    x = bn(x, training=training)
                if i == 0:
                    x = tf.nn.relu(x)
        elif self.slice_len == 65536:
            for filters, bn in self.final_layers:
                x = self.conv1d_transpose(x, filters, self.kernel_len, 4)
                if bn is not None:
                    x = bn(x, training=training)
                x = tf.nn.relu(x)
        return tf.nn.tanh(x)
def lrelu(inputs, alpha=0.2):
    return tf.maximum(alpha * inputs, inputs)

def apply_phaseshuffle(x, rad, pad_type='reflect'):
    b, x_len, nch = x.shape
    phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(tensor=x, paddings=[[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)
    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])
    return x

class WaveGANDiscriminator(Model):
    def __init__(self, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=0, slice_len=16384):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.phaseshuffle_rad = phaseshuffle_rad
        self.slice_len = slice_len

        self.conv0 = Conv1D(dim, kernel_size=kernel_len, strides=4, padding='same', name='downconv_0')
        self.conv1 = Conv1D(dim * 2, kernel_size=kernel_len, strides=4, padding='same', name='downconv_1')
        self.conv2 = Conv1D(dim * 4, kernel_size=kernel_len, strides=4, padding='same', name='downconv_2')
        self.conv3 = Conv1D(dim * 8, kernel_size=kernel_len, strides=4, padding='same', name='downconv_3')
        self.conv4 = Conv1D(dim * 16, kernel_size=kernel_len, strides=4, padding='same', name='downconv_4')

        # Conditionally add conv5 based on slice_len
        if self.slice_len == 32768:
            self.conv5 = Conv1D(dim * 32, kernel_size=kernel_len, strides=2, padding='same', name='downconv_5')
        elif self.slice_len == 65536:
            self.conv5 = Conv1D(dim * 32, kernel_size=kernel_len, strides=4, padding='same', name='downconv_5')
        else:
            self.conv5 = None

        # BatchNorm layers if used
        if use_batchnorm:
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()
            self.bn3 = BatchNormalization()
            self.bn4 = BatchNormalization()
            self.bn5 = BatchNormalization() if self.conv5 else None
        else:
            # Identity layers to avoid conditions in call
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
            self.bn3 = lambda x: x
            self.bn4 = lambda x: x
            self.bn5 = lambda x: x

        self.phaseshuffle = PhaseShuffle(phaseshuffle_rad)
        self.dense = Dense(1, name='output_dense')

    def call(self, x):
        batch_size = tf.shape(x)[0]

        # Layer 0
        x = self.conv0(x)
        x = lrelu(x)
        x = self.phaseshuffle(x)

        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = lrelu(x)
        x = self.phaseshuffle(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = lrelu(x)
        x = self.phaseshuffle(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = lrelu(x)
        x = self.phaseshuffle(x)

        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = lrelu(x)

        # Layer 5 if present
        if self.conv5 is not None:
            x = self.conv5(x)
            x = self.bn5(x)
            x = lrelu(x)

        # Flatten and output logit
        x = tf.reshape(x, [batch_size, -1])
        logits = self.dense(x)[:, 0]

        return logits
