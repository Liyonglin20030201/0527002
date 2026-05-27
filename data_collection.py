"""
模块一：数据采集模块
模拟手机加速度计采集洗衣机在不同工作状态下的振动数据。
实际使用时可替换为手机APP采集的真实数据。
"""

import numpy as np
import pandas as pd
import os


def generate_vibration_signal(state, duration_sec=10, sample_rate=100):
    """
    根据洗衣机状态生成模拟振动信号（三轴加速度）。

    参数:
        state: 工作状态 ('normal_wash', 'normal_spin', 'unbalance', 'bearing_fault', 'motor_fault')
        duration_sec: 采集时长（秒）
        sample_rate: 采样率（Hz）
    返回:
        numpy数组，shape=(n_samples, 3)，三轴加速度数据
    """
    n_samples = duration_sec * sample_rate
    t = np.linspace(0, duration_sec, n_samples)

    if state == 'normal_wash':
        freq = 2.0
        amplitude = 0.5
        noise_level = 0.1
        x = amplitude * np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(n_samples)
        y = amplitude * 0.8 * np.sin(2 * np.pi * freq * t + np.pi / 4) + noise_level * np.random.randn(n_samples)
        z = amplitude * 0.3 * np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(n_samples) + 9.8

    elif state == 'normal_spin':
        freq = 12.0
        amplitude = 2.0
        noise_level = 0.3
        x = amplitude * np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(n_samples)
        y = amplitude * np.cos(2 * np.pi * freq * t) + noise_level * np.random.randn(n_samples)
        z = amplitude * 0.5 * np.sin(2 * np.pi * freq * 2 * t) + noise_level * np.random.randn(n_samples) + 9.8

    elif state == 'unbalance':
        freq = 12.0
        amplitude = 4.5
        noise_level = 0.8
        mod_freq = 0.5
        modulation = 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        x = amplitude * modulation * np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(n_samples)
        y = amplitude * modulation * np.cos(2 * np.pi * freq * t) + noise_level * np.random.randn(n_samples)
        z = amplitude * 0.8 * np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(n_samples) + 9.8

    elif state == 'bearing_fault':
        freq = 12.0
        amplitude = 2.5
        noise_level = 1.2
        fault_freq = 45.0
        x = amplitude * np.sin(2 * np.pi * freq * t) + 1.5 * np.sin(2 * np.pi * fault_freq * t) + noise_level * np.random.randn(n_samples)
        y = amplitude * np.cos(2 * np.pi * freq * t) + 1.2 * np.sin(2 * np.pi * fault_freq * t + np.pi / 3) + noise_level * np.random.randn(n_samples)
        z = 0.8 * np.sin(2 * np.pi * fault_freq * t) + noise_level * np.random.randn(n_samples) + 9.8

    elif state == 'motor_fault':
        freq = 8.0
        amplitude = 1.8
        noise_level = 1.5
        x = amplitude * np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(n_samples)
        impulse_indices = np.random.choice(n_samples, size=n_samples // 50, replace=False)
        impulses = np.zeros(n_samples)
        impulses[impulse_indices] = np.random.uniform(3, 6, size=len(impulse_indices))
        x += impulses
        y = amplitude * 0.7 * np.sin(2 * np.pi * freq * t + np.pi / 6) + noise_level * np.random.randn(n_samples) + impulses * 0.5
        z = noise_level * np.random.randn(n_samples) + impulses * 0.3 + 9.8

    else:
        raise ValueError(f"未知状态: {state}")

    return np.column_stack([x, y, z])


def collect_dataset(n_samples_per_class=50, duration_sec=10, sample_rate=100, save_dir='data'):
    """
    生成完整数据集。

    参数:
        n_samples_per_class: 每种状态的样本数
        duration_sec: 每个样本的时长
        sample_rate: 采样率
        save_dir: 数据保存目录
    返回:
        保存路径
    """
    os.makedirs(save_dir, exist_ok=True)

    states = ['normal_wash', 'normal_spin', 'unbalance', 'bearing_fault', 'motor_fault']
    all_data = []

    for state in states:
        print(f"  采集 {state} 数据... ({n_samples_per_class} 个样本)")
        for i in range(n_samples_per_class):
            signal = generate_vibration_signal(state, duration_sec, sample_rate)
            sample_info = {
                'state': state,
                'sample_id': i,
                'duration_sec': duration_sec,
                'sample_rate': sample_rate,
                'data': signal
            }
            all_data.append(sample_info)

            filename = f"{state}_{i:03d}.csv"
            filepath = os.path.join(save_dir, filename)
            df = pd.DataFrame(signal, columns=['acc_x', 'acc_y', 'acc_z'])
            df.to_csv(filepath, index=False)

    metadata = pd.DataFrame([
        {'state': d['state'], 'sample_id': d['sample_id'], 'filename': f"{d['state']}_{d['sample_id']:03d}.csv"}
        for d in all_data
    ])
    metadata.to_csv(os.path.join(save_dir, 'metadata.csv'), index=False)

    print(f"  数据已保存到 {save_dir}/，共 {len(all_data)} 个样本")
    return save_dir


if __name__ == '__main__':
    print("=" * 50)
    print("洗衣机振动数据采集模块")
    print("=" * 50)
    collect_dataset(n_samples_per_class=50)
