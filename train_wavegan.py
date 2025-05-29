from __future__ import print_function

try:
  import cPickle as pickle
except:
  import pickle
from functools import reduce
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange

import loader
from wavegan import WaveGANGenerator, WaveGANDiscriminator


"""
  Trains a WaveGAN
"""
def train(fps, args):
    # Load and preprocess data
    x = list(loader.decode_extract_and_batch(
        fps,
        batch_size=args.train_batch_size,
        slice_len=args.data_slice_len,
        decode_fs=args.data_sample_rate,
        decode_num_channels=args.data_num_channels,
        decode_fast_wav=args.data_fast_wav,
        decode_parallel_calls=4,
        slice_randomize_offset=False if args.data_first_slice else True,
        slice_first_only=args.data_first_slice,
        slice_overlap_ratio=0. if args.data_first_slice else args.data_overlap_ratio,
        slice_pad_end=True if args.data_first_slice else args.data_pad_end,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=4096,
        prefetch_size=args.train_batch_size * 4,
        prefetch_gpu_num=args.data_prefetch_gpu_num)[:, :, 0])

    # Make z vector
    z = tf.random.uniform([args.train_batch_size, args.wavegan_latent_dim], -1., 1., dtype=tf.float32)

    # Define generator
    G = WaveGANGenerator(**args.wavegan_g_kwargs)
    G_z = G(z, training=True)
    if args.wavegan_genr_pp:
        G_z = tf.keras.layers.Conv1D(1, args.wavegan_genr_pp_len, use_bias=False, padding='same')(G_z)

    # Print generator summary
    print('-' * 80)
    print('Generator vars')
    nparams = sum(np.prod(v.shape) for v in G.trainable_variables)
    for v in G.trainable_variables:
        print('{} ({}): {}'.format(v.shape, np.prod(v.shape), v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

    # Summarize
    tf.summary.audio('x', x, args.data_sample_rate)
    tf.summary.audio('G_z', G_z, args.data_sample_rate)
    G_z_rms = tf.sqrt(tf.reduce_mean(tf.square(G_z[:, :, 0]), axis=1))
    x_rms = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, 0]), axis=1))
    tf.summary.histogram('x_rms_batch', x_rms)
    tf.summary.histogram('G_z_rms_batch', G_z_rms)
    tf.summary.scalar('x_rms', tf.reduce_mean(x_rms))
    tf.summary.scalar('G_z_rms', tf.reduce_mean(G_z_rms))

    # Define discriminator
    D = WaveGANDiscriminator(**args.wavegan_d_kwargs)
    D_x = D(x, training=True)
    D_G_z = D(G_z, training=True)

    # Print discriminator summary
    print('-' * 80)
    print('Discriminator vars')
    nparams = sum(np.prod(v.shape) for v in D.trainable_variables)
    for v in D.trainable_variables:
        print('{} ({}): {}'.format(v.shape, np.prod(v.shape), v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
    print('-' * 80)

    # Create loss
    if args.wavegan_loss == 'dcgan':
        fake = tf.zeros([args.train_batch_size], dtype=tf.float32)
        real = tf.ones([args.train_batch_size], dtype=tf.float32)
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_z, labels=real))
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_z, labels=fake))
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x, labels=real))
        D_loss /= 2.
    elif args.wavegan_loss == 'lsgan':
        G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
        D_loss = tf.reduce_mean((D_x - 1.) ** 2)
        D_loss += tf.reduce_mean(D_G_z ** 2)
        D_loss /= 2.
    elif args.wavegan_loss == 'wgan':
        G_loss = -tf.reduce_mean(D_G_z)
        D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
        D_clip_weights = [tf.clip_by_value(v, -0.01, 0.01) for v in D.trainable_variables]
    elif args.wavegan_loss == 'wgan-gp':
        G_loss = -tf.reduce_mean(D_G_z)
        D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
        alpha = tf.random.uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
        differences = G_z - x
        interpolates = x + (alpha * differences)
        D_interp = D(interpolates, training=True)
        LAMBDA = 10
        gradients = tf.gradients(D_interp, interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
        D_loss += LAMBDA * gradient_penalty
    else:
        raise NotImplementedError()

    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)

    # Create optimizers
    if args.wavegan_loss == 'dcgan':
        G_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        D_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    elif args.wavegan_loss == 'lsgan':
        G_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
        D_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    elif args.wavegan_loss == 'wgan':
        G_opt = tf.keras.optimizers.RMSprop(learning_rate=5e-5)
        D_opt = tf.keras.optimizers.RMSprop(learning_rate=5e-5)
    elif args.wavegan_loss == 'wgan-gp':
        G_opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        D_opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    else:
        raise NotImplementedError()

    # Training loop
    @tf.function
    def train_step():
        with tf.GradientTape(persistent=True) as tape:
            D_x = D(x, training=True)
            D_G_z = D(G_z, training=True)
            if args.wavegan_loss == 'dcgan':
                fake = tf.zeros([args.train_batch_size], dtype=tf.float32)
                real = tf.ones([args.train_batch_size], dtype=tf.float32)
                G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_z, labels=real))
                D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_z, labels=fake))
                D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x, labels=real))
                D_loss /= 2.
            elif args.wavegan_loss == 'lsgan':
                G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
                D_loss = tf.reduce_mean((D_x - 1.) ** 2)
                D_loss += tf.reduce_mean(D_G_z ** 2)
                D_loss /= 2.
            elif args.wavegan_loss == 'wgan':
                G_loss = -tf.reduce_mean(D_G_z)
                D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
            elif args.wavegan_loss == 'wgan-gp':
                G_loss = -tf.reduce_mean(D_G_z)
                D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
                alpha = tf.random.uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
                differences = G_z - x
                interpolates = x + (alpha * differences)
                D_interp = D(interpolates, training=True)
                LAMBDA = 10
                gradients = tf.gradients(D_interp, interpolates)[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
                D_loss += LAMBDA * gradient_penalty
            else:
                raise NotImplementedError()

        D_grad = tape.gradient(D_loss, D.trainable_variables)
        G_grad = tape.gradient(G_loss, G.trainable_variables)

        D_opt.apply_gradients(zip(D_grad, D.trainable_variables))
        G_opt.apply_gradients(zip(G_grad, G.trainable_variables))

        if args.wavegan_loss == 'wgan':
            for var, clip_var in zip(D.trainable_variables, D_clip_weights):
                var.assign(clip_var)

        return G_loss, D_loss

    # Run training
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = True

    with tf.compat.v1.train.MonitoredTrainingSession(
        config=config,
        checkpoint_dir=args.train_dir,
        save_checkpoint_secs=args.train_save_secs,
        save_summaries_secs=args.train_summary_secs) as sess:
        print('-' * 80)
        print('Training has started. Please use \'tensorboard --logdir={}\' to monitor.'.format(args.train_dir))
        while True:
            for _ in range(args.wavegan_disc_nupdates):
                G_loss_val, D_loss_val = train_step()
            print(f'G_loss: {G_loss_val.numpy()}, D_loss: {D_loss_val.numpy()}')


"""
  Creates and saves a MetaGraphDef for simple inference
  Tensors:
    'samp_z_n' int32 []: Sample this many latent vectors
    'samp_z' float32 [samp_z_n, latent_dim]: Resultant latent vectors
    'z:0' float32 [None, latent_dim]: Input latent vectors
    'flat_pad:0' int32 []: Number of padding samples to use when flattening batch to a single audio file
    'G_z:0' float32 [None, slice_len, 1]: Generated outputs
    'G_z_int16:0' int16 [None, slice_len, 1]: Same as above but quantizied to 16-bit PCM samples
    'G_z_flat:0' float32 [None, 1]: Outputs flattened into single audio file
    'G_z_flat_int16:0' int16 [None, 1]: Same as above but quantized to 16-bit PCM samples
  Example usage:
    import tensorflow as tf
    tf.reset_default_graph()

    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt-10000')

    z_n = graph.get_tensor_by_name('samp_z_n:0')
    _z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})

    z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
"""
def infer(args):
    infer_dir = os.path.join(args.train_dir, 'infer')
    if not os.path.isdir(infer_dir):
        os.makedirs(infer_dir)

    # Define the generator model
    G = WaveGANGenerator(**args.wavegan_g_kwargs)
    z = tf.keras.Input(shape=(args.wavegan_latent_dim,))
    G_z = G(z, training=False)
    if args.wavegan_genr_pp:
        G_z = tf.keras.layers.Conv1D(1, args.wavegan_genr_pp_len, use_bias=False, padding='same')(G_z)
    G_z = tf.identity(G_z, name='G_z')

    # Flatten batch
    nch = int(G_z.shape[-1])
    flat_pad = tf.keras.Input(shape=(), dtype=tf.int32, name='flat_pad')
    G_z_padded = tf.pad(G_z, paddings=[[0, 0], [0, flat_pad], [0, 0]])
    G_z_flat = tf.reshape(G_z_padded, [-1, nch], name='G_z_flat')

    # Encode to int16
    def float_to_int16(x, name=None):
        x_int16 = x * 32767.
        x_int16 = tf.clip_by_value(x_int16, -32767., 32767.)
        x_int16 = tf.cast(x_int16, tf.int16, name=name)
        return x_int16

    G_z_int16 = float_to_int16(G_z, name='G_z_int16')
    G_z_flat_int16 = float_to_int16(G_z_flat, name='G_z_flat_int16')

    # Create a model for inference
    inference_model = tf.keras.Model(inputs=[z, flat_pad], outputs=[G_z_int16, G_z_flat_int16])

    # Save the model
    inference_model.save(os.path.join(infer_dir, 'infer_model.h5'))

    # Reset graph (in case training afterwards)
    tf.keras.backend.clear_session()

"""
  Generates a preview audio file every time a checkpoint is saved
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wavwrite
from scipy.signal import freqz
import numpy as np
import tensorflow as tf

def preview(args):
    preview_dir = os.path.join(args.train_dir, 'preview')
    if not os.path.isdir(preview_dir):
        os.makedirs(preview_dir)

    # Load model
    model_path = os.path.join(args.train_dir, 'infer', 'infer_model.h5')
    G = tf.keras.models.load_model(model_path)

    # Generate or restore z_i and z_o
    z_fp = os.path.join(preview_dir, 'z.pkl')
    if os.path.exists(z_fp):
        with open(z_fp, 'rb') as f:
            _zs = pickle.load(f)
    else:
        # Sample z
        _zs = np.random.uniform(-1.0, 1.0, (args.preview_n, args.wavegan_latent_dim))
        # Save z
        with open(z_fp, 'wb') as f:
            pickle.dump(_zs, f)

    # Set up graph for generating preview images
    flat_pad = int(args.data_sample_rate / 2)
    feeds = {'z': _zs, 'flat_pad': flat_pad}
    fetches = {
        'G_z': G.output[0],
        'G_z_flat_int16': G.output[1]
    }
    if args.wavegan_genr_pp:
        pp_filter = G.get_layer('pp_filt').get_weights()[0][:, 0, 0]
        fetches['pp_filter'] = pp_filter

    # Summarize
    G_z = G.output[0]
    summaries = [
        tf.summary.audio('preview', tf.expand_dims(G_z, axis=0), args.data_sample_rate, max_outputs=1)
    ]
    summary_writer = tf.summary.create_file_writer(preview_dir)

    # PP Summarize
    if args.wavegan_genr_pp:
        pp_fp = tf.placeholder(tf.string, [])
        pp_bin = tf.io.read_file(pp_fp)
        pp_png = tf.image.decode_png(pp_bin)
        pp_summary = tf.summary.image('pp_filt', tf.expand_dims(pp_png, axis=0))

    # Loop, waiting for checkpoints
    ckpt_fp = None
    while True:
        latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
        if latest_ckpt_fp != ckpt_fp:
            print('Preview: {}'.format(latest_ckpt_fp))
            with tf.compat.v1.Session() as sess:
                saver.restore(sess, latest_ckpt_fp)
                _fetches = sess.run(fetches, feeds)
                _step = _fetches['step']
            preview_fp = os.path.join(preview_dir, '{}.wav'.format(str(_step).zfill(8)))
            wavwrite(preview_fp, args.data_sample_rate, _fetches['G_z_flat_int16'])
            with summary_writer.as_default():
                tf.summary.experimental.write_raw_pb(_fetches['summaries'].SerializeToString(), step=_step)
            if args.wavegan_genr_pp:
                w, h = freqz(_fetches['pp_filter'])
                fig = plt.figure()
                plt.title('Digital filter frequency response')
                ax1 = fig.add_subplot(111)
                plt.plot(w, 20 * np.log10(abs(h)), 'b')
                plt.ylabel('Amplitude [dB]', color='b')
                plt.xlabel('Frequency [rad/sample]')
                ax2 = ax1.twinx()
                angles = np.unwrap(np.angle(h))
                plt.plot(w, angles, 'g')
                plt.ylabel('Angle (radians)', color='g')
                plt.grid()
                plt.axis('tight')
                _pp_fp = os.path.join(preview_dir, '{}_ppfilt.png'.format(str(_step).zfill(8)))
                plt.savefig(_pp_fp)
                with tf.compat.v1.Session() as sess:
                    _summary = sess.run(pp_summary, {pp_fp: _pp_fp})
                    with summary_writer.as_default():
                        tf.summary.experimental.write_raw_pb(_summary.SerializeToString(), step=_step)
            print('Done')
            ckpt_fp = latest_ckpt_fp
        time.sleep(1)

"""
  Computes inception score every time a checkpoint is saved
"""
import os
import pickle
import numpy as np
import tensorflow as tf
from scipy.stats import entropy

def incept(args):
    incept_dir = os.path.join(args.train_dir, 'incept')
    if not os.path.isdir(incept_dir):
        os.makedirs(incept_dir)

    # Load GAN model
    gan_model_path = os.path.join(args.train_dir, 'infer', 'infer_model.h5')
    G = tf.keras.models.load_model(gan_model_path)

    # Generate or restore latents
    z_fp = os.path.join(incept_dir, 'z.pkl')
    if os.path.exists(z_fp):
        with open(z_fp, 'rb') as f:
            _zs = pickle.load(f)
    else:
        _zs = np.random.uniform(-1.0, 1.0, (args.incept_n, args.wavegan_latent_dim))
        with open(z_fp, 'wb') as f:
            pickle.dump(_zs, f)

    # Load classifier model
    classifier_model_path = args.incept_ckpt_fp
    classifier = tf.keras.models.load_model(classifier_model_path)

    # Create summaries
    summary_writer = tf.summary.create_file_writer(incept_dir)

    # Loop, waiting for checkpoints
    ckpt_fp = None
    _best_score = 0.
    while True:
        latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
        if latest_ckpt_fp != ckpt_fp:
            print('Incept: {}'.format(latest_ckpt_fp))
            sess = tf.compat.v1.Session()
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())
            sess.run(tf.compat.v1.report_uninitialized_variables())

            # Restore GAN model
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, latest_ckpt_fp)

            # Run GAN to generate samples
            _G_zs = []
            for i in range(0, args.incept_n, 100):
                _G_zs.append(sess.run(G.output[0], feed_dict={G.input[0]: _zs[i:i+100]}))
            _G_zs = np.concatenate(_G_zs, axis=0)

            # Run classifier on generated samples
            _preds = []
            for i in range(0, args.incept_n, 100):
                _preds.append(classifier.predict(_G_zs[i:i+100]))
            _preds = np.concatenate(_preds, axis=0)

            # Split into k groups
            _incept_scores = []
            split_size = args.incept_n // args.incept_k
            for i in range(args.incept_k):
                _split = _preds[i * split_size:(i + 1) * split_size]
                _kl = _split * (np.log(_split) - np.log(np.expand_dims(np.mean(_split, 0), 0)))
                _kl = np.mean(np.sum(_kl, 1))
                _incept_scores.append(np.exp(_kl))

            _incept_mean, _incept_std = np.mean(_incept_scores), np.std(_incept_scores)

            # Summarize
            with summary_writer.as_default():
                tf.summary.scalar('incept_mean', _incept_mean)
                tf.summary.scalar('incept_std', _incept_std)
                summary_writer.flush()

            # Save best score
            if _incept_mean > _best_score:
                score_saver = tf.compat.v1.train.Saver()
                score_saver.save(sess, os.path.join(incept_dir, 'best_score'), global_step=int(_incept_mean))
                _best_score = _incept_mean

            sess.close()
            print('Done')
            ckpt_fp = latest_ckpt_fp
        time.sleep(1)

if __name__ == '__main__':
    import argparse
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
    parser.add_argument('train_dir', type=str,
                        help='Training directory')
    data_args = parser.add_argument_group('Data')
    data_args.add_argument('--data_dir', type=str,
                          help='Data directory containing _only_ audio files to load')
    data_args.add_argument('--data_sample_rate', type=int,
                          help='Number of audio samples per second')
    data_args.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536],
                          help='Number of audio samples per slice (maximum generation length)')
    data_args.add_argument('--data_num_channels', type=int,
                          help='Number of audio channels to generate (for >2, must match that of data)')
    data_args.add_argument('--data_overlap_ratio', type=float,
                          help='Overlap ratio [0, 1) between slices')
    data_args.add_argument('--data_first_slice', action='store_true', dest='data_first_slice',
                          help='If set, only use the first slice each audio example')
    data_args.add_argument('--data_pad_end', action='store_true', dest='data_pad_end',
                          help='If set, use zero-padded partial slices from the end of each audio file')
    data_args.add_argument('--data_normalize', action='store_true', dest='data_normalize',
                          help='If set, normalize the training examples')
    data_args.add_argument('--data_fast_wav', action='store_true', dest='data_fast_wav',
                          help='If your data is comprised of standard WAV files (16-bit signed PCM or 32-bit float), use this flag to decode audio using scipy (faster) instead of librosa')
    data_args.add_argument('--data_prefetch_gpu_num', type=int,
                          help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')
    wavegan_args = parser.add_argument_group('WaveGAN')
    wavegan_args.add_argument('--wavegan_latent_dim', type=int,
                              help='Number of dimensions of the latent space')
    wavegan_args.add_argument('--wavegan_kernel_len', type=int,
                              help='Length of 1D filter kernels')
    wavegan_args.add_argument('--wavegan_dim', type=int,
                              help='Dimensionality multiplier for model of G and D')
    wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',
                              help='Enable batchnorm')
    wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,
                              help='Number of discriminator updates per generator update')
    wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
                              help='Which GAN loss to use')
    wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn'],
                              help='Generator upsample strategy')
    wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', dest='wavegan_genr_pp',
                              help='If set, use post-processing filter')
    wavegan_args.add_argument('--wavegan_genr_pp_len', type=int,
                              help='Length of post-processing filter for DCGAN')
    wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int,
                              help='Radius of phase shuffle operation')
    train_args = parser.add_argument_group('Train')
    train_args.add_argument('--train_batch_size', type=int,
                            help='Batch size')
    train_args.add_argument('--train_save_secs', type=int,
                            help='How often to save model')
    train_args.add_argument('--train_summary_secs', type=int,
                            help='How often to report summaries')
    preview_args = parser.add_argument_group('Preview')
    preview_args.add_argument('--preview_n', type=int,
                              help='Number of samples to preview')
    incept_args = parser.add_argument_group('Incept')
    incept_args.add_argument('--incept_metagraph_fp', type=str,
                                help='Inference model for inception score')
    incept_args.add_argument('--incept_ckpt_fp', type=str,
                                help='Checkpoint for inference model')
    incept_args.add_argument('--incept_n', type=int,
                              help='Number of generated examples to test')
    incept_args.add_argument('--incept_k', type=int,
                              help='Number of groups to test')
    parser.set_defaults(
        data_dir=None,
        data_sample_rate=16000,
        data_slice_len=16384,
        data_num_channels=1,
        data_overlap_ratio=0.,
        data_first_slice=False,
        data_pad_end=False,
        data_normalize=False,
        data_fast_wav=False,
        data_prefetch_gpu_num=0,
        wavegan_latent_dim=100,
        wavegan_kernel_len=25,
        wavegan_dim=64,
        wavegan_batchnorm=False,
        wavegan_disc_nupdates=5,
        wavegan_loss='wgan-gp',
        wavegan_genr_upsample='zeros',
        wavegan_genr_pp=False,
        wavegan_genr_pp_len=512,
        wavegan_disc_phaseshuffle=2,
        train_batch_size=128,
        train_save_secs=60 * 15,
        train_summary_secs=360,
        preview_n=32,
        incept_metagraph_fp='./eval/inception/infer.meta',
        incept_ckpt_fp='./eval/inception/best_acc-103005',
        incept_n=5000,
        incept_k=10)
    args = parser.parse_args()

    # Make train dir
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)

    # Save args
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Make model kwarg dicts
    setattr(args, 'wavegan_g_kwargs', {
        'slice_len': args.data_slice_len,
        'nch': args.data_num_channels,
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
        'upsample': args.wavegan_genr_upsample
    })
    setattr(args, 'wavegan_d_kwargs', {
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
        'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
    })

    if args.mode == 'train':
        fps = glob.glob(os.path.join(args.data_dir, '*'))
        if len(fps) == 0:
            raise Exception('Did not find any audio files in specified directory')
        print('Found {} audio files in specified directory'.format(len(fps)))
        train(fps, args)
    elif args.mode == 'preview':
        preview(args)
    elif args.mode == 'incept':
        incept(args)
    elif args.mode == 'infer':
        infer(args)
    else:
        raise NotImplementedError()
