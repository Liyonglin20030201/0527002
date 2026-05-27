"""
模块二：特征提取模块
从原始三轴加速度数据中提取时域和频域特征。
"""

import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq


def extract_time_domain_features(signal):
    """
    提取单轴信号的时域特征。

    参数:
        signal: 一维数组，单轴加速度数据
    返回:
        字典，包含各时域特征
    """
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['rms'] = np.sqrt(np.mean(signal ** 2))
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    features['peak_to_peak'] = np.max(signal) - np.min(signal)
    features['skewness'] = stats.skew(signal)
    features['kurtosis'] = stats.kurtosis(signal)
    features['crest_factor'] = np.max(np.abs(signal)) / features['rms'] if features['rms'] > 0 else 0
    features['shape_factor'] = features['rms'] / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0
    n = len(signal)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal - np.mean(signal)))) > 0)
    features['zero_crossing_rate'] = zero_crossings / n
    features['energy'] = np.sum(signal ** 2) / n
    return features


def extract_frequency_domain_features(signal, sample_rate=100):
    """
    提取单轴信号的频域特征。

    参数:
        signal: 一维数组，单轴加速度数据
        sample_rate: 采样率（Hz）
    返回:
        字典，包含各频域特征
    """
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / sample_rate)

    positive_mask = xf > 0
    xf_pos = xf[positive_mask]
    magnitude = 2.0 / n * np.abs(yf[positive_mask])

    features = {}
    features['dominant_freq'] = xf_pos[np.argmax(magnitude)]
    features['dominant_magnitude'] = np.max(magnitude)
    features['spectral_centroid'] = np.sum(xf_pos * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
    features['spectral_spread'] = np.sqrt(
        np.sum(((xf_pos - features['spectral_centroid']) ** 2) * magnitude) / np.sum(magnitude)
    ) if np.sum(magnitude) > 0 else 0
    features['spectral_energy'] = np.sum(magnitude ** 2)
    features['spectral_entropy'] = -np.sum(
        (magnitude / np.sum(magnitude)) * np.log2(magnitude / np.sum(magnitude) + 1e-12)
    ) if np.sum(magnitude) > 0 else 0

    bands = [(0, 5), (5, 15), (15, 30), (30, 50)]
    for low, high in bands:
        band_mask = (xf_pos >= low) & (xf_pos < high)
        features[f'band_energy_{low}_{high}'] = np.sum(magnitude[band_mask] ** 2)

    return features


def extract_features_from_sample(data, sample_rate=100):
    """
    从一个完整的三轴加速度样本中提取所有特征。

    参数:
        data: numpy数组，shape=(n_samples, 3)
        sample_rate: 采样率
    返回:
        一维特征向量（字典）
    """
    axes = ['x', 'y', 'z']
    all_features = {}

    for i, axis in enumerate(axes):
        signal = data[:, i]

        time_features = extract_time_domain_features(signal)
        for key, value in time_features.items():
            all_features[f'{axis}_{key}'] = value

        freq_features = extract_frequency_domain_features(signal, sample_rate)
        for key, value in freq_features.items():
            all_features[f'{axis}_{key}'] = value

    magnitude = np.sqrt(np.sum(data ** 2, axis=1))
    mag_time = extract_time_domain_features(magnitude)
    for key, value in mag_time.items():
        all_features[f'mag_{key}'] = value
    mag_freq = extract_frequency_domain_features(magnitude, sample_rate)
    for key, value in mag_freq.items():
        all_features[f'mag_{key}'] = value

    return all_features


if __name__ == '__main__':
    from data_collection import generate_vibration_signal

    print("=" * 50)
    print("特征提取模块演示")
    print("=" * 50)

    for state in ['normal_wash', 'normal_spin', 'bearing_fault']:
        signal = generate_vibration_signal(state, duration_sec=5)
        features = extract_features_from_sample(signal)
        print(f"\n{state} 提取了 {len(features)} 个特征")
        print(f"  主频(X轴): {features['x_dominant_freq']:.2f} Hz")
        print(f"  RMS(X轴): {features['x_rms']:.4f}")
        print(f"  峰峰值(X轴): {features['x_peak_to_peak']:.4f}")
