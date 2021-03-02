import os.path as op
from typing import BinaryIO, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import uniform_filter1d
import librosa
import parselmouth

def get_waveform(
    path_or_fp: Union[str, BinaryIO], normalization=True
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit mono-channel WAV or FLAC.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): Normalize values to [-1, 1] (Default: True)
    """
    if isinstance(path_or_fp, str):
        ext = op.splitext(op.basename(path_or_fp))[1]
        if ext not in {".flac", ".wav"}:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLAC file")

    waveform, sample_rate = sf.read(path_or_fp, dtype="float32")
    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    return waveform, sample_rate


def _get_kaldi_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.fbank import FbankOptions, Fbank
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_torchaudio_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torch
        import torchaudio.compliance.kaldi as ta_kaldi

        waveform = torch.from_numpy(waveform).unsqueeze(0)
        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate
        )
        return features.numpy()
    except ImportError:
        return None


def get_fbank(path_or_fp: Union[str, BinaryIO], n_bins=80) -> np.ndarray:
    """Get mel-filter bank features via PyKaldi or TorchAudio. Prefer PyKaldi
    (faster CPP implementation) to TorchAudio (Python implementation). Note that
    Kaldi/TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized."""
    sound, sample_rate = get_waveform(path_or_fp, normalization=False)

    features = _get_kaldi_fbank(sound, sample_rate, n_bins)
    if features is None:
        features = _get_torchaudio_fbank(sound, sample_rate, n_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )

    return features

def get_speech_features(path_or_fp: Union[str, BinaryIO], data_cfg, max_frames, n_speech_features) -> np.ndarray:
    sound = parselmouth.Sound(path_or_fp)

    speech_features = np.empty(shape=(n_speech_features, max_frames))
    feat_offset = 0
    if data_cfg.pitch['use_pitch'] or data_cfg.pitch['use_pov'] or data_cfg.pitch['use_delta_pitch']:
       pitch = get_pitch(sound, data_cfg.pitch['time_step'], data_cfg.pitch['min_f0'], data_cfg.pitch['max_f0'])
       pitch, pov, delta_pitch = post_process_pitch(pitch, max_frames)

       if data_cfg.pitch['use_pitch']:
            speech_features[feat_offset] = pitch
            feat_offset += 1
       if data_cfg.pitch['use_pov']:
            speech_features[feat_offset] = pov
            feat_offset += 1
       if data_cfg.pitch['use_delta_pitch']:
            speech_features[feat_offset] = delta_pitch
            feat_offset += 1

    return speech_features.transpose()

def get_pitch(sound, time_step, min_f0, max_f0) -> np.ndarray:
    pitch = sound.to_pitch(time_step, min_f0, max_f0)
    pitch_values = pitch.selected_array['frequency']
    return pitch_values

def post_process_pitch(pitch, max_frames):
    # Pad or trim pitch depending on the number of frames in the spectral features.
    pitch = pad_trim(pitch, max_frames)

    # Interpolate unvoiced regions.
    pitch, nonzero_idx = interpolate_zeros(pitch)

    pitch = np.log(pitch + 1e-10)

    # Compute delta pitch.
    delta_pitch = librosa.feature.delta(pitch, order=1)

    # Apply mean substraction to smooth out unexpected peaks, with a window size of 151 frames, as suggested by Kaldi.
    pitch = uniform_filter1d(pitch, size=151, mode='reflect')

    # Get probability-of-voicing (POV) vector. Keep it between -1 and 1 for stability.
    pov = np.full(pitch.shape, -1.0)
    pov[nonzero_idx] = 1.0

    return pitch, pov, delta_pitch

def interpolate_zeros(signal):
    '''Interpolates zero regions. Returns the interpolated signal and the non-zero indexes.'''
    x = np.arange(len(signal))
    idx = np.nonzero(signal)

    if np.count_nonzero(signal):
        # If boundaries of the signal sequence have zero values, interpolate with first and last non-zero values.
        first_nonzero_value, last_nonzero_value = signal[idx][0], signal[idx][-1]
        f = interp1d(x[idx], signal[idx], bounds_error=False, fill_value=(first_nonzero_value, last_nonzero_value))
        signal = f(x)

        # Add a little noise to signal values.
        mean_noise = 0.0
        std_noise = 0.5
        noise = np.random.normal(mean_noise, std_noise, len(signal))
        signal += noise

    return signal, idx

def pad_trim(buffer, max_frames):
    if len(buffer) < max_frames:
        pad = max_frames - len(buffer)
        buffer = np.concatenate((buffer, buffer[-pad:]), axis=0)
    elif len(buffer) > max_frames:
        trim = len(buffer) - max_frames
        buffer = buffer[:-trim]

    return buffer