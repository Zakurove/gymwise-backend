import pandas as pd
import numpy as np
from django.conf import settings
import os

def load_and_preprocess_data(file_name):
    file_path = os.path.join(settings.BASE_DIR, 'data', file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} does not exist in the data directory.")
    
    df = pd.read_csv(file_path)
    return preprocess_data(df)

def preprocess_data(df):
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    processed_df = processed_df.fillna(0)
    
    # Convert categorical variables to numeric if they exist
    categorical_mappings = {
        'gender': {'male': 0, 'female': 1},
        'time_of_day': {'morning': 0, 'afternoon': 1, 'evening': 2},
    }
    
    for col, mapping in categorical_mappings.items():
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].map(mapping).fillna(0).astype(int)
    
    # Handle date-related features
    date_column = next((col for col in processed_df.columns if 'date' in col.lower()), None)
    if date_column:
        processed_df[date_column] = pd.to_datetime(processed_df[date_column], errors='coerce')
        processed_df['days_since_last_visit'] = (pd.Timestamp.now() - processed_df[date_column]).dt.days
        processed_df['month'] = processed_df[date_column].dt.month
        processed_df['is_ramadan'] = processed_df['month'].isin([9, 10])
        processed_df['is_summer'] = processed_df['month'].isin([6, 7, 8])
    
    # Handle numeric columns
    numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
    
    # Create visit frequency if possible
    if 'id' in processed_df.columns and date_column:
        processed_df['visit_frequency'] = processed_df.groupby('id')[date_column].transform('count')
    
    # Ensure core fields exist, create them if they don't
    core_fields = ['email', 'name', 'churn_risk', 'churn_probability']
    for field in core_fields:
        if field not in processed_df.columns:
            if field in ['churn_risk', 'churn_probability']:
                processed_df[field] = 0  # Default values
            else:
                processed_df[field] = ''  # Empty string for non-numeric fields
    
    return processed_df