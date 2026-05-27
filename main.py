"""
洗衣机故障检测系统 - 主程序
基于手机加速度计振动数据的智能故障诊断
"""

import sys
import os
import time

from data_collection import collect_dataset
from preprocessing import load_and_preprocess_data, prepare_training_data
from model_training import (
    train_random_forest, evaluate_model, cross_validate_model,
    get_feature_importance, save_model
)
from prediction import WashingMachineDiagnostor, run_diagnosis_demo


def main():
    print()
    print("=" * 60)
    print("     洗衣机故障检测系统 v1.0")
    print("     基于手机加速度计振动数据 + 随机森林分类")
    print("=" * 60)
    print()

    # ============ 步骤1: 数据采集 ============
    print("[步骤 1/5] 数据采集")
    print("-" * 40)
    if not os.path.exists('data/metadata.csv'):
        print("  生成模拟振动数据...")
        collect_dataset(n_samples_per_class=50, duration_sec=10, sample_rate=100)
    else:
        print("  数据已存在，跳过采集。")
    print()

    # ============ 步骤2&3: 特征提取 + 预处理 ============
    print("[步骤 2/5] 特征提取")
    print("[步骤 3/5] 数据预处理")
    print("-" * 40)
    print("  加载数据并提取特征...")
    X, y, feature_names = load_and_preprocess_data(
        data_dir='data', window_size=500, overlap=0.5, sample_rate=100
    )

    print("  标准化与划分数据集...")
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_training_data(X, y)
    print()

    # ============ 步骤4: 模型训练 ============
    print("[步骤 4/5] 模型训练")
    print("-" * 40)
    print("  训练随机森林分类器...")
    start_time = time.time()
    model = train_random_forest(X_train, y_train)
    train_time = time.time() - start_time
    print(f"  训练耗时: {train_time:.2f} 秒")

    print("\n  模型评估:")
    evaluate_model(model, X_test, y_test, label_encoder)
    cross_validate_model(model, X_train, y_train)
    get_feature_importance(model, feature_names)
    save_model(model)
    print()

    # ============ 步骤5: 预测与展示 ============
    print("[步骤 5/5] 故障预测演示")
    print("-" * 40)
    run_diagnosis_demo()

    print("\n" + "=" * 60)
    print("  系统就绪！")
    print("  使用方法:")
    print("    1. 将手机放在洗衣机顶部")
    print("    2. 用加速度计APP记录振动数据(CSV格式)")
    print("    3. 运行: python prediction.py 或在代码中调用:")
    print("       from prediction import WashingMachineDiagnostor")
    print("       d = WashingMachineDiagnostor()")
    print("       result = d.predict_from_file('your_data.csv')")
    print("       d.display_result(result)")
    print("=" * 60)


if __name__ == '__main__':
    main()
