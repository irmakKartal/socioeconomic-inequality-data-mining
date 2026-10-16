"""
Machine Learning Models Module: Linear Regression & K-Means Clustering.
Implements algorithms used in the Nişantaşı University Data Mining Research Project.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score


def train_linear_regression(df, x_col="poverty_rate_pct", y_col="gender_wage_gap_pct"):
    """
    Fits an Ordinary Least Squares (OLS) Linear Regression model between two features.

    Returns:
        dict: Fitted model, evaluation metrics (R2, MSE, RMSE, MAE), slope, intercept.
    """
    clean_df = df[[x_col, y_col]].dropna()
    X = clean_df[[x_col]].values
    y = clean_df[y_col].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    # Generate smooth line for visualization
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_range)

    return {
        "model": model,
        "x_col": x_col,
        "y_col": y_col,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "x_range": x_range.flatten(),
        "y_line": y_line,
        "equation": f"y = {slope:.4f} * x + {intercept:.4f}",
    }


def train_kmeans_clustering(df, feature_cols, n_clusters=3, random_state=42):
    """
    Performs K-Means Clustering on normalized socio-economic indicators.

    Returns:
        dict: Scaler, fitted KMeans model, clustered DataFrame, silhouette score, inertia.
    """
    sub_df = df[feature_cols].dropna().copy()
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(sub_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_matrix)

    sil_score = None
    if 2 <= n_clusters < len(sub_df):
        sil_score = silhouette_score(scaled_matrix, cluster_labels)

    result_df = df.copy()
    result_df["Cluster"] = [f"Cluster {c}" for c in cluster_labels]
    result_df["Cluster_ID"] = cluster_labels

    # Compute cluster profiles (means of each feature per cluster)
    profiles = result_df.groupby("Cluster")[feature_cols].mean().round(3)

    return {
        "model": kmeans,
        "scaler": scaler,
        "clustered_df": result_df,
        "silhouette_score": sil_score,
        "inertia": kmeans.inertia_,
        "cluster_profiles": profiles,
        "feature_cols": feature_cols,
    }


def compute_elbow_curve(df, feature_cols, k_range=range(2, 7)):
    """
    Computes inertia and silhouette scores for K in k_range to identify the optimal elbow.
    """
    sub_df = df[feature_cols].dropna()
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(sub_df)

    inertias = []
    silhouettes = []
    ks = list(k_range)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_matrix)
        inertias.append(km.inertia_)
        sil = silhouette_score(scaled_matrix, km.labels_)
        silhouettes.append(sil)

    return {"k_values": ks, "inertias": inertias, "silhouettes": silhouettes}
