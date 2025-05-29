from __future__ import print_function
import numpy as np
from scipy.io import wavfile
import librosa
import tensorflow as tf
import logging
import os

# Configure logging
logging.basicConfig(level=logging.WARNING)

def decode_audio(fp, fs=None, num_channels=1, normalize=False, fast_wav=False):
    """Decodes audio file paths into 32-bit floating point vectors.
    
    Args:
        fp: Audio file path (string or TensorFlow tensor).
        fs: If specified, resamples decoded audio to this rate.
        num_channels: Number of channels for decoded audio files.
        normalize: If true, normalizes the audio waveforms.
        fast_wav: Assume fp is a standard WAV file (PCM 16-bit or float 32-bit).
    
    Returns:
        A np.float32 array containing the audio samples at specified sample rate.
        Returns empty array ([0, 1, num_channels]) on error.
    """
    # Convert TensorFlow tensor to string if necessary
    if isinstance(fp, tf.Tensor):
        fp = tf.compat.as_str_any(fp)
    
    # Check if the file exists
    if not os.path.exists(fp):
        logging.error(f"File not found: {fp}")
        return np.zeros([0, 1, num_channels], dtype=np.float32)
    
    # Attempt to read the audio file
    try:
        if fast_wav:
            # Read with scipy.io.wavfile (fast)
            try:
                _fs, _wav = wavfile.read(fp)
                if fs is not None and fs != _fs:
                    raise NotImplementedError('Scipy cannot resample audio.')
                if _wav.dtype == np.int16:
                    _wav = _wav.astype(np.float32) / 32768.0
                elif _wav.dtype == np.float32:
                    _wav = np.copy(_wav)
                else:
                    raise NotImplementedError('Scipy cannot process atypical WAV files.')
            except Exception as e:
                logging.warning(f"Scipy failed, falling back to librosa for {fp}: {e}")
                _wav, _fs = librosa.core.load(fp, sr=fs, mono=False)
        else:
            # Decode with librosa (supports multiple formats)
            _wav, _fs = librosa.core.load(fp, sr=fs, mono=False)
    except Exception as e:
        logging.error(f"Failed to read {fp}: {e}")
        return np.zeros([0, 1, num_channels], dtype=np.float32)
    
    # Process audio shape
    if _wav.ndim == 2:
        _wav = np.swapaxes(_wav, 0, 1)
    
    assert _wav.dtype == np.float32
    
    # Reshape to [nsamps, 1, nch]
    if _wav.ndim == 1:
        nsamps = _wav.shape[0]
        nch = 1
    else:
        nsamps, nch = _wav.shape
    _wav = np.reshape(_wav, [nsamps, 1, nch])
    
    # Handle channel mismatch
    if nch != num_channels:
        if num_channels == 1:
            _wav = np.mean(_wav, axis=2, keepdims=True)
        elif nch == 1 and num_channels == 2:
            _wav = np.concatenate([_wav, _wav], axis=2)
        else:
            raise ValueError('Number of audio channels must be 1 or 2')
    
    # Normalize if requested
    if normalize:
        factor = np.max(np.abs(_wav))
        if factor > 0:
            _wav /= factor
    
    return _wav

# The rest of the code (decode_extract_and_batch) remains unchanged
def decode_extract_and_batch(
    fps,
    batch_size,
    slice_len,
    decode_fs,
    decode_num_channels,
    decode_normalize=True,
    decode_fast_wav=False,
    decode_parallel_calls=1,
    slice_randomize_offset=False,
    slice_first_only=False,
    slice_overlap_ratio=0,
    slice_pad_end=False,
    repeat=False,
    shuffle=False,
    shuffle_buffer_size=None,
    prefetch_size=None,
    prefetch_gpu_num=None):
    """Decodes audio file paths into mini-batches of samples.
    Args:
        fps: List of audio file paths.
        batch_size: Number of items in the batch.
        slice_len: Length of the slice sequences in samples or feature timesteps.
        decode_fs: (Re-)sample rate for decoded audio files.
        decode_num_channels: Number of channels for decoded audio files.
        decode_normalize: If false, do not normalize audio waveforms.
        decode_fast_wav: If true, uses scipy to decode standard wav files.
        decode_parallel_calls: Number of parallel decoding threads.
        slice_randomize_offset: If true, randomize starting position for slice.
        slice_first_only: If true, only use first slice from each audio file.
        slice_overlap_ratio: Ratio of overlap between adjacent slices.
        slice_pad_end: If true, allows zero-padded examples from the end of each audio file.
        repeat: If true (for training), continuously iterate through the dataset.
        shuffle: If true (for training), buffer and shuffle the slice sequences.
        shuffle_buffer_size: Number of examples to queue up before grabbing a batch.
        prefetch_size: Number of examples to prefetch from the queue.
        prefetch_gpu_num: If specified, prefetch examples to GPU.
    Returns:
        A tuple of np.float32 tensors representing audio waveforms.
          audio: [batch_size, slice_len, 1, nch]
    """
    # Create dataset of filepaths
    dataset = tf.data.Dataset.from_tensor_slices(fps)
    
    # Shuffle all filepaths every epoch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(fps))
    
    # Repeat
    if repeat:
        dataset = dataset.repeat()
    
    def _decode_audio_shaped(fp):
        _decode_audio_closure = lambda _fp: decode_audio(
            _fp,
            fs=decode_fs,
            num_channels=decode_num_channels,
            normalize=decode_normalize,
            fast_wav=decode_fast_wav)
        audio = tf.py_function(
            _decode_audio_closure,
            [fp],
            tf.float32)
        audio.set_shape([None, 1, decode_num_channels])
        return audio
    
    # Decode audio
    dataset = dataset.map(
        _decode_audio_shaped,
        num_parallel_calls=decode_parallel_calls)
    
    # Parallel
    def _slice(audio):
        # Calculate hop size
        if slice_overlap_ratio < 0:
            raise ValueError('Overlap ratio must be greater than 0')
        slice_hop = int(round(slice_len * (1. - slice_overlap_ratio)) + 1e-4)
        if slice_hop < 1:
            raise ValueError('Overlap ratio too high')
        
        # Randomize starting phase:
        if slice_randomize_offset:
            start = tf.random.uniform([], maxval=slice_len, dtype=tf.int32)
            audio = audio[start:]
        
        # Extract slice sequences
        audio_slices = tf.signal.frame(
            audio,
            slice_len,
            slice_hop,
            pad_end=slice_pad_end,
            pad_value=0,
            axis=0)
        
        # Only use first slice if requested
        if slice_first_only:
            audio_slices = audio_slices[:1]
        
        return audio_slices
    
    def _slice_dataset_wrapper(audio):
        audio_slices = _slice(audio)
        return tf.data.Dataset.from_tensor_slices(audio_slices)
    
    # Extract parallel slice sequences from both audio and features
    dataset = dataset.flat_map(_slice_dataset_wrapper)
    
    # Shuffle examples
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    # Make batches
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefetch a number of batches
    if prefetch_size is not None:
        dataset = dataset.prefetch(prefetch_size)
        if prefetch_gpu_num is not None and prefetch_gpu_num >= 0:
            dataset = dataset.apply(
                tf.data.experimental.prefetch_to_device(
                    '/device:GPU:{}'.format(prefetch_gpu_num)))
    
    # Get tensors
    return next(iter(dataset))
