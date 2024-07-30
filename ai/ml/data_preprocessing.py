import pandas as pd
import numpy as np
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

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(0)
    
    # Convert categorical variables to numeric
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    df['time_of_day'] = df['time_of_day'].map({'morning': 0, 'afternoon': 1, 'evening': 2})
    
    # Create new features
    df['days_since_last_visit'] = (pd.Timestamp.now() - pd.to_datetime(df['date'])).dt.days
    df['visit_frequency'] = df.groupby('id')['date'].transform('count')
    
    # Add seasonal features
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['is_ramadan'] = df['month'].isin([9, 10])  # Assuming Ramadan is typically in the 9th and 10th months
    df['is_summer'] = df['month'].isin([6, 7, 8])
    
    return df

def preprocess_scenario_data(df, scenario_params):
    df = preprocess_data(df)
    
    # Apply scenario parameters
    if 'membership_price_change' in scenario_params:
        df['membership_price'] *= (1 + scenario_params['membership_price_change'])
    
    if 'new_classes' in scenario_params:
        df['available_classes'] += scenario_params['new_classes']
    
    if 'gym_hours_change' in scenario_params:
        df['gym_hours'] += scenario_params['gym_hours_change']
    
    if 'marketing_intensity' in scenario_params:
        df['marketing_score'] = df['marketing_score'] * scenario_params['marketing_intensity']
    
    # Add more scenario parameter applications as needed
    
    return df