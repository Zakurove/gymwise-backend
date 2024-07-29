import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def perform_clustering(X_train_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X_train_scaled)

def analyze_segments(df, clusters):
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df = df.iloc[:len(clusters)]  # Ensure df has the same length as clusters
    df['Cluster'] = clusters
    segment_profile = df.groupby('Cluster').mean()
    churn_by_segment = df.groupby('Cluster')['Churn'].mean()
    return segment_profile, churn_by_segment