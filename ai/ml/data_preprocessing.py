import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from django.conf import settings
import os

def load_and_preprocess_data(file_name):
    file_path = os.path.join(settings.BASE_DIR, 'data', file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} does not exist in the data directory.")
    
    df = pd.read_csv(file_path)
    
    binary_cols = ['gender', 'Near_Location', 'Partner', 'Promo_friends', 'Phone', 'Group_visits', 'Churn']
    df[binary_cols] = df[binary_cols].astype(bool)
    
    return df

def split_and_scale_data(df, target_column='Churn'):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler