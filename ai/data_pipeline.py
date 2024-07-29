import pandas as pd
from django.conf import settings
from .models import Member, MemberActivity
import logging

logger = logging.getLogger(__name__)

def load_data_for_institution(institution_id):
    try:
        members = Member.objects.filter(institution_id=institution_id).values()
        activities = MemberActivity.objects.filter(member__institution_id=institution_id).values()
        
        df_members = pd.DataFrame.from_records(members)
        df_activities = pd.DataFrame.from_records(activities)
        
        # Merge member and activity data
        df = pd.merge(df_members, df_activities, left_on='id', right_on='member_id', how='left')
        
        return df
    except Exception as e:
        logger.error(f"Error loading data for institution {institution_id}: {str(e)}")
        return None

def preprocess_data(df):
    try:
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
        
        # Select relevant features
        features = ['age', 'gender', 'membership_duration', 'visit_frequency', 'days_since_last_visit', 
                    'churn_probability', 'is_ramadan', 'is_summer']
        
        return df[features]
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        return None

def get_preprocessed_data_for_institution(institution_id):
    raw_data = load_data_for_institution(institution_id)
    if raw_data is not None:
        preprocessed_data = preprocess_data(raw_data)
        return preprocessed_data
    return None