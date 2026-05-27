"""
模块五：预测与展示模块
加载训练好的模型，对新的振动数据进行故障预测。
"""

import numpy as np
import pandas as pd
import joblib
import os

from feature_extraction import extract_features_from_sample
from preprocessing import segment_signal


STATE_DESCRIPTIONS = {
    'normal_wash': {'name': '正常洗涤', 'icon': '[OK]', 'action': '洗衣机运行正常，无需维修。'},
    'normal_spin': {'name': '正常脱水', 'icon': '[OK]', 'action': '洗衣机运行正常，无需维修。'},
    'unbalance': {'name': '不平衡故障', 'icon': '[!!]', 'action': '衣物分布不均，建议重新放置衣物后再运行。'},
    'bearing_fault': {'name': '轴承故障', 'icon': '[XX]', 'action': '轴承可能磨损，建议尽快联系维修人员更换轴承。'},
    'motor_fault': {'name': '电机故障', 'icon': '[XX]', 'action': '电机运行异常，建议立即停机并联系专业维修。'},
}


class WashingMachineDiagnostor:
    """洗衣机故障诊断器。"""

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')

        for path in [model_path, scaler_path, encoder_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"模型文件不存在: {path}\n请先运行 main.py 训练模型。")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)

    def predict_from_file(self, filepath, sample_rate=100):
        """
        从CSV文件预测故障类型。

        参数:
            filepath: CSV文件路径（三列：acc_x, acc_y, acc_z）
            sample_rate: 采样率
        返回:
            预测结果字典
        """
        data = pd.read_csv(filepath).values
        return self.predict_from_array(data, sample_rate)

    def predict_from_array(self, data, sample_rate=100):
        """
        从numpy数组预测故障类型。

        参数:
            data: numpy数组，shape=(n_samples, 3)
            sample_rate: 采样率
        返回:
            预测结果字典
        """
        segments = segment_signal(data, window_size=min(500, len(data)), overlap=0.5)

        if not segments:
            segments = [data]

        predictions = []
        probabilities = []

        for segment in segments:
            features = extract_features_from_sample(segment, sample_rate)
            feature_vector = np.array([list(features.values())])
            feature_scaled = self.scaler.transform(feature_vector)
            pred = self.model.predict(feature_scaled)[0]
            prob = self.model.predict_proba(feature_scaled)[0]
            predictions.append(pred)
            probabilities.append(prob)

        from collections import Counter
        vote_counts = Counter(predictions)
        final_prediction = vote_counts.most_common(1)[0][0]
        final_label = self.label_encoder.inverse_transform([final_prediction])[0]

        avg_prob = np.mean(probabilities, axis=0)
        confidence = avg_prob[final_prediction]

        result = {
            'state': final_label,
            'confidence': confidence,
            'description': STATE_DESCRIPTIONS.get(final_label, {}),
            'all_probabilities': dict(zip(self.label_encoder.classes_, avg_prob)),
            'n_segments': len(segments),
        }
        return result

    def display_result(self, result):
        """格式化显示预测结果。"""
        desc = result['description']

        print("\n" + "=" * 50)
        print(f"  {desc['icon']} 诊断结果: {desc['name']}")
        print("=" * 50)
        print(f"  置信度: {result['confidence'] * 100:.1f}%")
        print(f"  建议: {desc['action']}")
        print(f"  (基于 {result['n_segments']} 个数据窗口的综合判断)")
        print()
        print("  各状态概率:")
        for state, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
            bar = '#' * int(prob * 30)
            state_name = STATE_DESCRIPTIONS.get(state, {}).get('name', state)
            print(f"    {state_name:10s} | {bar:30s} | {prob * 100:.1f}%")
        print("=" * 50)


def run_diagnosis_demo():
    """运行诊断演示。"""
    from data_collection import generate_vibration_signal

    print("\n" + "=" * 50)
    print("  洗衣机故障诊断系统 - 演示模式")
    print("=" * 50)

    diagnostor = WashingMachineDiagnostor()

    test_states = ['normal_spin', 'unbalance', 'bearing_fault', 'motor_fault']

    for state in test_states:
        print(f"\n>>> 模拟输入: {state} 状态的振动数据")
        test_data = generate_vibration_signal(state, duration_sec=10)
        result = diagnostor.predict_from_array(test_data)
        diagnostor.display_result(result)


if __name__ == '__main__':
    run_diagnosis_demo()
