import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_k_values(X, k_range):
    sse = []
    silhouette_scores = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        sse.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    return sse, silhouette_scores


def plot_elbow(k_range, sse, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('K value')
    plt.ylabel('SSE')
    plt.title('Elbow Method on Iris Dataset')
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_silhouette(k_range, silhouette_scores, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('K value')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score on Iris Dataset')
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def best_cluster_label_mapping(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {cluster_label: true_label for true_label, cluster_label in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping[label] for label in y_pred])
    accuracy = np.mean(mapped_pred == y_true)
    return mapping, mapped_pred, accuracy, cm


def plot_final_clusters(X, labels, centers, save_path):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    centers_2d = pca.transform(centers)

    plt.figure(figsize=(8, 5))
    for cluster_id in range(centers.shape[0]):
        plt.scatter(
            X_2d[labels == cluster_id, 0],
            X_2d[labels == cluster_id, 1],
            label=f'Cluster {cluster_id}',
            alpha=0.75,
        )

    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        marker='X',
        s=250,
        label='Centroids',
    )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Final K-Means Clustering Result (K=3)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    output_dir = 'kmeans_iris_output'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 载入数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print('=== Iris 数据集基本信息 ===')
    print(f'样本数: {X.shape[0]}')
    print(f'特征数: {X.shape[1]}')
    print(f'特征名称: {feature_names}')
    print(f'类别名称: {target_names.tolist()}')
    print()

    k_range = range(2, 11)
    sse, silhouette_scores = evaluate_k_values(X, k_range)

    print('=== 不同 K 值的评价结果 ===')
    print('K\tSSE\t\tSilhouette Score')
    for k, inertia, score in zip(k_range, sse, silhouette_scores):
        print(f'{k}\t{inertia:.4f}\t\t{score:.4f}')
    print()

    plot_elbow(k_range, sse, os.path.join(output_dir, 'elbow_method.png'))
    plot_silhouette(k_range, silhouette_scores, os.path.join(output_dir, 'silhouette_score.png'))

    best_k = 3
    print(f'最终选择的最佳 K 值: {best_k}')
    print('理由: 肘部法则在 K=3 附近出现明显拐点；虽然 K=2 的轮廓系数更高，')
    print('但会把 versicolor 和 virginica 更容易混到一起，而数据集本身包含 3 个品种，因此最终选 K=3。')
    print()

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    cluster_labels = final_model.fit_predict(X)
    centers = final_model.cluster_centers_

    print('=== 聚类中心（4个特征） ===')
    print(centers)
    print()

    mapping, mapped_pred, accuracy, cm = best_cluster_label_mapping(y, cluster_labels)
    print('=== 聚类标签到真实类别的最佳映射 ===')
    print(mapping)
    print()
    print('=== 混淆矩阵（行是真实类别，列是聚类标签） ===')
    print(cm)
    print()
    print(f'映射后的聚类准确率: {accuracy:.4f}')
    print()

    plot_final_clusters(X, cluster_labels, centers, os.path.join(output_dir, 'final_clusters_pca.png'))

    print('图像已保存到以下目录:')
    print(os.path.abspath(output_dir))
    print('- elbow_method.png')
    print('- silhouette_score.png')
    print('- final_clusters_pca.png')


if __name__ == '__main__':
    main()
