"""
模块三：数据预处理模块
数据切分、特征提取与标准化处理。
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

from feature_extraction import extract_features_from_sample


def segment_signal(data, window_size=500, overlap=0.5):
    """
    将长时间序列切分为固定长度的窗口。

    参数:
        data: numpy数组，shape=(n_samples, 3)
        window_size: 窗口大小（采样点数）
        overlap: 重叠比例 (0~1)
    返回:
        切分后的片段列表
    """
    step = int(window_size * (1 - overlap))
    segments = []
    for start in range(0, len(data) - window_size + 1, step):
        segment = data[start:start + window_size]
        segments.append(segment)
    return segments


def load_and_preprocess_data(data_dir='data', window_size=500, overlap=0.5, sample_rate=100):
    """
    加载原始数据，切分窗口，提取特征。

    参数:
        data_dir: 数据目录
        window_size: 窗口大小
        overlap: 重叠比例
        sample_rate: 采样率
    返回:
        X: 特征矩阵
        y: 标签数组
        feature_names: 特征名列表
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)

    all_features = []
    all_labels = []

    print(f"  正在处理 {len(metadata)} 个样本文件...")

    for _, row in metadata.iterrows():
        filepath = os.path.join(data_dir, row['filename'])
        data = pd.read_csv(filepath).values
        state = row['state']

        segments = segment_signal(data, window_size, overlap)

        for segment in segments:
            features = extract_features_from_sample(segment, sample_rate)
            all_features.append(features)
            all_labels.append(state)

    feature_df = pd.DataFrame(all_features)
    X = feature_df.values
    feature_names = feature_df.columns.tolist()
    y = np.array(all_labels)

    print(f"  特征提取完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    return X, y, feature_names


def prepare_training_data(X, y, test_size=0.2, random_state=42, save_dir='models'):
    """
    标准化特征并划分训练/测试集。

    参数:
        X: 特征矩阵
        y: 标签数组
        test_size: 测试集比例
        random_state: 随机种子
        save_dir: 模型保存目录
    返回:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    os.makedirs(save_dir, exist_ok=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nan_mask = np.isnan(X_scaled).any(axis=1) | np.isinf(X_scaled).any(axis=1)
    if nan_mask.any():
        print(f"  警告: 移除 {nan_mask.sum()} 个包含异常值的样本")
        X_scaled = X_scaled[~nan_mask]
        y_encoded = y_encoded[~nan_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))

    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    print(f"  类别: {list(label_encoder.classes_)}")

    return X_train, X_test, y_train, y_test, scaler, label_encoder


if __name__ == '__main__':
    print("=" * 50)
    print("数据预处理模块")
    print("=" * 50)

    print("\n[1] 加载并提取特征...")
    X, y, feature_names = load_and_preprocess_data()

    print("\n[2] 标准化与数据划分...")
    X_train, X_test, y_train, y_test, scaler, le = prepare_training_data(X, y)

    print(f"\n特征维度: {X_train.shape[1]}")
    print(f"类别映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")
