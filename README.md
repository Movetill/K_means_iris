# K-Means Iris Clustering

This project applies the **K-Means clustering algorithm** to the classic **Iris dataset** and evaluates the clustering performance using the **Elbow Method** and **Silhouette Score**. It also visualizes the final clustering result with **PCA dimensionality reduction**.

## Project Overview

The main goal of this project is to:

- Load the Iris dataset
- Cluster the data using K-Means
- Compare different values of `K`
- Evaluate clustering quality using:
  - **SSE (Sum of Squared Errors)**
  - **Silhouette Score**
- Select the most appropriate number of clusters
- Visualize the final clustering result in 2D using PCA
- Compare clustering labels with the true labels for analysis

> Note: K-Means is an **unsupervised learning algorithm**, so the true labels are **not used during training**. They are only used afterward for evaluation and comparison.

## Dataset

This project uses the **Iris dataset** from `scikit-learn`.

- **Number of samples:** 150
- **Number of features:** 4
- **Classes:** 3
  - Setosa
  - Versicolor
  - Virginica

Features:

- Sepal length
- Sepal width
- Petal length
- Petal width

## File Structure

```bash
.
├── kmeans_iris.py
└── kmeans_iris_output/
    ├── elbow_method.png
    ├── silhouette_score.png
    └── final_clusters_pca.png
```

## Requirements

Install the required Python libraries before running the project:

```bash
pip install numpy matplotlib scikit-learn scipy
```

## How to Run

Run the following command in the project directory:

```bash
python kmeans_iris.py
```

## Output

After running the script, the program will:

1. Print the basic information of the Iris dataset
2. Calculate SSE and Silhouette Score for different values of `K`
3. Save the following figures:

- `elbow_method.png`
- `silhouette_score.png`
- `final_clusters_pca.png`

These images will be stored in the `kmeans_iris_output` folder.

## Main Methods Used

### 1. K-Means Clustering

K-Means partitions the dataset into `K` clusters by minimizing the sum of squared distances between samples and their cluster centers.

### 2. Elbow Method

The Elbow Method is used to observe how SSE changes as `K` increases. A clear turning point helps estimate a suitable number of clusters.

### 3. Silhouette Score

The Silhouette Score evaluates how well samples fit within their assigned clusters compared with other clusters.

### 4. PCA Visualization

Since the Iris dataset has 4 features, PCA is used to reduce the data to 2 dimensions for visualization.

## Result Summary

In this project, `K=3` is selected as the final clustering number based on:

- the elbow trend in SSE,
- the clustering structure of the Iris dataset,
- and the fact that the dataset itself contains 3 flower categories.

The final result shows that K-Means can distinguish **Setosa** very well, while **Versicolor** and **Virginica** may have some overlap.

## Notes

- K-Means is sensitive to the initial cluster centers.
- Different initializations may produce slightly different results.
- In this project, `random_state=42` and multiple initializations (`n_init=20`) are used to improve stability.

## Author

**Movetill**
