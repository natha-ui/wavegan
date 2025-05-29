from scipy.io.wavfile import read as wavread
import numpy as np
import tensorflow as tf
import sys

def decode_audio(fp, fs=None, num_channels=1, normalize=False, fast_wav=False):
    """Decodes audio file paths into 32-bit floating point vectors.

    Args:
        fp: Audio file path.
        fs: If specified, resamples decoded audio to this rate.
        num_channels: Target number of channels (1 for mono, 2 for stereo)
        normalize: If true, normalizes audio to [-1, 1] range
        fast_wav: Assume fp is a standard WAV file (PCM 16-bit or float 32-bit)

    Returns:
        A np.float32 array containing the audio samples at specified sample rate.
    """
    #print("loading audio file %s" % fp)
    if fast_wav:
        # Read with scipy wavread (fast).
        _fs, _wav = wavread(fp)
        if fs is not None and fs != _fs:
            raise NotImplementedError('Scipy cannot resample audio.')
        if _wav.dtype == np.int16:
            _wav = _wav.astype(np.float32)
            _wav /= 32768.
        elif _wav.dtype == np.float32:
            _wav = np.copy(_wav)
        else:
            raise NotImplementedError('Scipy cannot process atypical WAV files.')
    else:
        # Decode with librosa load (slow but supports file formats like mp3).
        import librosa
        try:
            _wav, _fs = librosa.core.load(fp, sr=fs, mono=False)
        except:
            print("LOADER WARNING: Failed on %s" % fp)
            _wav,_fs=librosa.core.load('/home/matt/datasets/drumsamples/Korg_KorgS3_KorgS3Set2_Fx70.wav',sr=fs,mono=False)
        if _wav.ndim == 2:
            _wav = np.swapaxes(_wav,0,1)

    assert _wav.dtype == np.float32

    # At this point, _wav is np.float32 either [nsamps,] or [nsamps, nch].
    # We want [nsamps, 1, nch] to mimic 2D shape of spectral feats.
    if _wav.ndim == 1:
        nsamps = _wav.shape[0]
        nch = 1
    else:
        nsamps, nch = _wav.shape
    _wav = np.reshape(_wav, [nsamps, 1, nch])
 
    # Average (mono) or expand (stereo) channels
    if nch != num_channels:
        if num_channels == 1:
            _wav = np.mean(_wav, 2, keepdims=True)
        elif nch == 1 and num_channels == 2:
            _wav = np.concatenate([_wav, _wav], axis=2)
        else:
            raise ValueError('Number of audio channels not equal to num specified')

    if normalize:
        factor = np.max(np.abs(_wav))
        if factor > 0:
            _wav /= factor

    return _wav


def decode_extract_and_batch(
    fps,
    batch_size,
    slice_len,
    decode_fs,
    decode_num_channels,
    decode_normalize=True,
    decode_fast_wav=False,
    decode_parallel_calls=tf.data.AUTOTUNE,
    slice_randomize_offset=False,
    slice_first_only=False,
    slice_overlap_ratio=0,
    slice_pad_end=False,
    repeat=False,
    shuffle=False,
    shuffle_buffer_size=None,
    prefetch_size=tf.data.AUTOTUNE,
    prefetch_gpu_num=None):
    """Decodes audio file paths into mini-batches of samples (Keras 3/TF2 compatible)."""
    # Create dataset of filepaths
    dataset = tf.data.Dataset.from_tensor_slices(fps)

    # Shuffle filepaths
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(fps))

    # Repeat indefinitely if needed
    if repeat:
        dataset = dataset.repeat()

    # Decode audio function
    def _decode_audio_wrapper(fp):
        audio = tf.py_function(
            func=lambda x: decode_audio(
                x.numpy().decode('utf-8'),
                fs=decode_fs,
                num_channels=decode_num_channels,
                normalize=decode_normalize,
                fast_wav=decode_fast_wav),
            inp=[fp],
            Tout=tf.float32
        )
        audio.set_shape([None, 1, decode_num_channels])
        return audio

    # Decode audio files
    dataset = dataset.map(
        _decode_audio_wrapper,
        num_parallel_calls=decode_parallel_calls
    )

    # Slice audio function
    def _slice_audio(audio):
        # Calculate hop size
        slice_hop = int(round(slice_len * (1.0 - slice_overlap_ratio)))
        if slice_hop < 1:
            raise ValueError('Overlap ratio too high')

        # Random starting offset
        if slice_randomize_offset:
            max_start = tf.maximum(tf.shape(audio)[0] - slice_len, 0)
            start = tf.random.uniform([], 0, max_start + 1, dtype=tf.int32)
            audio = audio[start:start + slice_len]
            return tf.expand_dims(audio, axis=0)

        # Extract slices
        slices = tf.signal.frame(
            audio,
            slice_len,
            slice_hop,
            pad_end=slice_pad_end,
            pad_value=0,
            axis=0
        )
        return slices if not slice_first_only else slices[:1]

    # Process slices
    def _process_audio(audio):
        slices = _slice_audio(audio)
        return tf.data.Dataset.from_tensor_slices(slices)

    dataset = dataset.flat_map(_process_audio)

    # Shuffle slices
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Device prefetching
    if prefetch_gpu_num is not None and prefetch_gpu_num >= 0:
        dataset = dataset.apply(
            tf.data.experimental.prefetch_to_device(f'/device:GPU:{prefetch_gpu_num}'))
    elif prefetch_size is not None:
        dataset = dataset.prefetch(prefetch_size)

    return dataset
