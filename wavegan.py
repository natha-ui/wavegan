import tensorflow as tf
from tensorflow.keras.layers import Conv1DTranspose, Conv1D, BatchNormalization, Dense, Reshape
from tensorflow.keras.models import Model

def conv1d_transpose(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample='zeros'):
    
    if upsample == 'zeros':
        return Conv1DTranspose(filters, kernel_size=kernel_width, strides=stride, padding=padding)(tf.expand_dims(inputs, axis=1))[:, 0]
    elif upsample == 'nn':
        batch_size = tf.shape(inputs)[0]
        _, w, nch = inputs.shape
        x = inputs
        x = tf.expand_dims(x, axis=1)
        x = tf.image.resize(x, [1, w * stride], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = x[:, 0]
        return Conv1D(filters, kernel_size=kernel_width, strides=1, padding=padding)(x)
    else:
        raise NotImplementedError

def WaveGANGenerator(
    z,
    slice_len=16384,
    nch=1,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=False):
    
    assert slice_len in [16384, 32768, 65536]
    batch_size = tf.shape(z)[0]
    
    if use_batchnorm:
        batchnorm = lambda x: BatchNormalization()(x)
    else:
        batchnorm = lambda x: x
    
    # FC and reshape for convolution
    # [100] -> [16, 1024]
    dim_mul = 16 if slice_len == 16384 else 32
    output = z
    with tf.name_scope('z_project'):
        output = Dense(4096 * dim * dim_mul)(output)
        output = Reshape((16, dim * dim_mul))(output)
        output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2
    
    # Layer 0
    # [16, 1024] -> [64, 512]
    with tf.name_scope('upconv_0'):
        output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
        output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2
    
    # Layer 1
    # [64, 512] -> [256, 256]
    with tf.name_scope('upconv_1'):
        output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
        output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2
    
    # Layer 2
    # [256, 256] -> [1024, 128]
    with tf.name_scope('upconv_2'):
        output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
        output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2
    
    # Layer 3
    # [1024, 128] -> [4096, 64]
    with tf.name_scope('upconv_3'):
        output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
        output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2
    
    if slice_len == 16384:
        # Layer 4
        # [4096, 64] -> [16384, nch]
        with tf.name_scope('upconv_4'):
            output = conv1d_transpose(output, nch, kernel_len, 4, upsample=upsample)
        output = tf.nn.tanh(output)
    elif slice_len == 32768:
        # Layer 4
        # [4096, 128] -> [16384, 64]
        with tf.name_scope('upconv_4'):
            output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf.nn.relu(output)
        # Layer 5
        # [16384, 64] -> [32768, nch]
        with tf.name_scope('upconv_5'):
            output = conv1d_transpose(output, nch, kernel_len, 2, upsample=upsample)
        output = tf.nn.tanh(output)
    elif slice_len == 65536:
        # Layer 4
        # [4096, 128] -> [16384, 64]
        with tf.name_scope('upconv_4'):
            output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf.nn.relu(output)
        # Layer 5
        # [16384, 64] -> [65536, nch]
        with tf.name_scope('upconv_5'):
            output = conv1d_transpose(output, nch, kernel_len, 4, upsample=upsample)
        output = tf.nn.tanh(output)
    
    # Automatically update batchnorm moving averages every time G is used during training
    if train and use_batchnorm:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
        if slice_len == 16384:
            assert len(update_ops) == 10
        else:
            assert len(update_ops) == 12
        with tf.control_dependencies(update_ops):
            output = tf.identity(output)
    
    return output

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

def WaveGANDiscriminator(
    x,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0):
    
    batch_size = tf.shape(x)[0]
    slice_len = int(x.shape[1])
    
    if use_batchnorm:
        batchnorm = lambda x: BatchNormalization()(x)
    else:
        batchnorm = lambda x: x
    
    if phaseshuffle_rad > 0:
        phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
    else:
        phaseshuffle = lambda x: x
    
    # Layer 0
    # [16384, 1] -> [4096, 64]
    output = x
    with tf.name_scope('downconv_0'):
        output = Conv1D(dim, kernel_size=kernel_len, strides=4, padding='SAME')(output)
    output = lrelu(output)
    output = phaseshuffle(output)
    
    # Layer 1
    # [4096, 64] -> [1024, 128]
    with tf.name_scope('downconv_1'):
        output = Conv1D(dim * 2, kernel_size=kernel_len, strides=4, padding='SAME')(output)
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)
    
    # Layer 2
    # [1024, 128] -> [256, 256]
    with tf.name_scope('downconv_2'):
        output = Conv1D(dim * 4, kernel_size=kernel_len, strides=4, padding='SAME')(output)
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)
    
    # Layer 3
    # [256, 256] -> [64, 512]
    with tf.name_scope('downconv_3'):
        output = Conv1D(dim * 8, kernel_size=kernel_len, strides=4, padding='SAME')(output)
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)
    
    # Layer 4
    # [64, 512] -> [16, 1024]
    with tf.name_scope('downconv_4'):
        output = Conv1D(dim * 16, kernel_size=kernel_len, strides=4, padding='SAME')(output)
        output = batchnorm(output)
    output = lrelu(output)
    
    if slice_len == 32768:
        # Layer 5
        # [32, 1024] -> [16, 2048]
        with tf.name_scope('downconv_5'):
            output = Conv1D(dim * 32, kernel_size=kernel_len, strides=2, padding='SAME')(output)
            output = batchnorm(output)
        output = lrelu(output)
    elif slice_len == 65536:
        # [64, 1024] -> [16, 2048]
        with tf.name_scope('downconv_5'):
            output = Conv1D(dim * 32, kernel_size=kernel_len, strides=4, padding='SAME')(output)
            output = batchnorm(output)
        output = lrelu(output)
    
    # Flatten
    output = tf.reshape(output, [batch_size, -1])
    
    # Connect to single logit
    with tf.name_scope('output'):
        output = Dense(1)(output)[:, 0]
    
    # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training
    return output
