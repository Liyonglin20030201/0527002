"""
模块四：模型训练模块
使用随机森林训练洗衣机故障分类器。
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
import os
import matplotlib.pyplot as plt


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    训练随机森林分类器。

    参数:
        X_train: 训练特征
        y_train: 训练标签
        n_estimators: 树的数量
        random_state: 随机种子
    返回:
        训练好的模型
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_encoder, save_dir='models'):
    """
    评估模型性能。

    参数:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        label_encoder: 标签编码器
        save_dir: 保存目录
    返回:
        评估指标字典
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_names = label_encoder.classes_

    print(f"\n  模型准确率: {accuracy:.4f}")
    print(f"\n  分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    print("  混淆矩阵:")
    print(f"  {'':>15}", end='')
    for name in class_names:
        print(f"{name:>15}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>15}", end='')
        for val in row:
            print(f"{val:>15}", end='')
        print()

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=100)
    plt.close()

    return {'accuracy': accuracy, 'confusion_matrix': cm}


def cross_validate_model(model, X_train, y_train, cv=5):
    """
    交叉验证评估模型稳定性。
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"\n  {cv}折交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores


def get_feature_importance(model, feature_names, top_n=15, save_dir='models'):
    """
    获取并可视化特征重要性。
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n  Top {top_n} 重要特征:")
    for rank, idx in enumerate(indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s} = {importances[idx]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=100)
    plt.close()

    return importances, indices


def save_model(model, save_dir='models', filename='random_forest_model.pkl'):
    """保存训练好的模型。"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    joblib.dump(model, path)
    print(f"\n  模型已保存到: {path}")
    return path


if __name__ == '__main__':
    from preprocessing import load_and_preprocess_data, prepare_training_data

    print("=" * 50)
    print("模型训练模块")
    print("=" * 50)

    print("\n[1] 加载数据...")
    X, y, feature_names = load_and_preprocess_data()

    print("\n[2] 预处理...")
    X_train, X_test, y_train, y_test, scaler, le = prepare_training_data(X, y)

    print("\n[3] 训练随机森林...")
    model = train_random_forest(X_train, y_train)

    print("\n[4] 评估模型...")
    evaluate_model(model, X_test, y_test, le)

    print("\n[5] 交叉验证...")
    cross_validate_model(model, X_train, y_train)

    print("\n[6] 特征重要性...")
    get_feature_importance(model, feature_names)

    print("\n[7] 保存模型...")
    save_model(model)
