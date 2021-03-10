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
        if data_cfg.pitch['pitch_path'] or data_cfg.pitch['pov_path'] or data_cfg.pitch['delta_pitch_path']:
            pass
        else:
            pitch = get_pitch(sound, data_cfg.pitch['time_step'], data_cfg.pitch['min_f0'], data_cfg.pitch['max_f0'])
            pitch, pov, delta_pitch = post_process_pitch(pitch, max_frames)

        if data_cfg.pitch['use_pitch']:
            if data_cfg.pitch['pitch_path']:
                pitch = get_feature_from_npy(data_cfg.pitch['pitch_path'], path_or_fp)
            speech_features[feat_offset] = pitch
            feat_offset += 1
        if data_cfg.pitch['use_pov']:
            if data_cfg.pitch['pov_path']:
                pov = get_feature_from_npy(data_cfg.pitch['pov_path'], path_or_fp)
            speech_features[feat_offset] = pov
            feat_offset += 1
        if data_cfg.pitch['use_delta_pitch']:
            if data_cfg.pitch['delta_pitch_path']:
                delta_pitch = get_feature_from_npy(data_cfg.pitch['delta_pitch_path'], path_or_fp)
            speech_features[feat_offset] = delta_pitch
            feat_offset += 1

    if data_cfg.voice_quality['use_jitter_local'] or data_cfg.voice_quality['use_shimmer_local']:
        if data_cfg.voice_quality['jitter_local_path'] or data_cfg.voice_quality['shimmer_local_path']:
            pass
        else:
            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", data_cfg.pitch['min_f0'], data_cfg.pitch['max_f0'])

        if data_cfg.voice_quality['use_jitter_local']:
            if data_cfg.voice_quality['jitter_local_path']:
                jitter_local = get_feature_from_npy(data_cfg.voice_quality['jitter_local_path'], path_or_fp)
            else:
                jitter_local = get_jitter(sound, point_process, max_frames, jitter_type="Get jitter (local)",  data_cfg=data_cfg)
            speech_features[feat_offset] = jitter_local
            feat_offset += 1
        if data_cfg.voice_quality['use_shimmer_local']:
            if data_cfg.voice_quality['shimmer_local_path']:
                shimmer_local = get_feature_from_npy(data_cfg.voice_quality['shimmer_local_path'], path_or_fp)
            else:
                shimmer_local = get_shimmer(sound, point_process, max_frames, shimmer_type="Get shimmer (local)", data_cfg=data_cfg)
            speech_features[feat_offset] = shimmer_local
            feat_offset += 1

    if data_cfg.pitch['random_feats'] > 0:
        for i in range(data_cfg.pitch['random_feats']):
            speech_features[feat_offset] = np.random.rand(max_frames)*10.0
            feat_offset += 1

    return speech_features.transpose()

def get_feature_from_npy(base_path, audio_path):
    """Given a base path and the file name from the audio path, 
    it reads the feature stored in the .npy file."""
    file_name = f"{op.basename(audio_path).split('.')[0]}.npy"
    npy_path = op.join(base_path, file_name)
    return np.load(npy_path)

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

def get_jitter(sound, point_process, max_frames, jitter_type, data_cfg):
    win_length = data_cfg.voice_quality['win_length']
    win_hop = data_cfg.voice_quality['win_hop']
    period_floor = data_cfg.voice_quality['period_floor']
    period_ceiling = data_cfg.voice_quality['period_ceiling']
    max_period_factor = data_cfg.voice_quality['max_period_factor']
    filter_frames = data_cfg.voice_quality['filter_frames']
    length = len(sound)/sound.get_sampling_frequency()
    start_time_s = 0.0

    start_times = np.arange(start_time_s, length - win_length, win_hop)
    end_times = np.arange(start_time_s + win_length, length, win_hop)
    times = np.column_stack((start_times, end_times)).tolist()

    get_segment_jitter = lambda times : parselmouth.praat.call(point_process, jitter_type, times[0], times[1], period_floor, period_ceiling, max_period_factor)
    jitter_list = list(map(get_segment_jitter, times))

    jitter_array = np.asarray(jitter_list)
    jitter_array = pad_trim(jitter_array, max_frames)
    jitter_array = jitter_array*100.0

    jitter_array[np.isnan(jitter_array)] = 0.0
    jitter_array, _ = interpolate_zeros(jitter_array)

    jitter_array = uniform_filter1d(jitter_array, size=filter_frames, mode='reflect')

    return jitter_array

def get_shimmer(sound, point_process, max_frames, shimmer_type, data_cfg):
    win_length = data_cfg.voice_quality['win_length']
    win_hop = data_cfg.voice_quality['win_hop']
    period_floor = data_cfg.voice_quality['period_floor']
    period_ceiling = data_cfg.voice_quality['period_ceiling']
    max_period_factor = data_cfg.voice_quality['max_period_factor']
    max_amplitude_factor = data_cfg.voice_quality['max_amplitude_factor']
    filter_frames = data_cfg.voice_quality['filter_frames']
    length = len(sound)/sound.get_sampling_frequency()
    start_time_s = 0.0

    start_times = np.arange(start_time_s, length - win_length, win_hop)
    end_times = np.arange(start_time_s + win_length, length, win_hop)
    times = np.column_stack((start_times, end_times)).tolist()

    get_segment_shimmer = lambda times : parselmouth.praat.call([sound, point_process], shimmer_type, times[0], times[1], period_floor, period_ceiling, max_period_factor, max_amplitude_factor)
    shimmer_list = list(map(get_segment_shimmer, times))

    shimmer_array = np.asarray(shimmer_list)
    shimmer_array = pad_trim(shimmer_array, max_frames)
    shimmer_array = shimmer_array*100.0

    shimmer_array[np.isnan(shimmer_array)] = 0.0
    shimmer_array, _ = interpolate_zeros(shimmer_array)

    shimmer_array = uniform_filter1d(shimmer_array, size=filter_frames, mode='reflect')

    return shimmer_array

def interpolate_zeros(signal):
    '''Interpolates zero regions. Returns the interpolated signal and the non-zero indexes.'''
    x = np.arange(len(signal))
    idx = np.nonzero(signal)

    if np.count_nonzero(signal) and len(idx[0]) > 1:
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