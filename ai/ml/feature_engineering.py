import numpy as np

def engineer_features(df):
    df_engineered = df.copy()

    # Create Contract_length if Contract_period exists
    if 'Contract_period' in df_engineered.columns:
        df_engineered['Contract_length'] = df_engineered['Contract_period'].map({1: 'Short', 6: 'Medium', 12: 'Long'})
    
    # Create Time_left_ratio if both Month_to_end_contract and Contract_period exist
    if 'Month_to_end_contract' in df_engineered.columns and 'Contract_period' in df_engineered.columns:
        df_engineered['Time_left_ratio'] = df_engineered['Month_to_end_contract'] / df_engineered['Contract_period']
    
    # Create Avg_additional_charges_per_month if both Avg_additional_charges_total and Lifetime exist
    if 'Avg_additional_charges_total' in df_engineered.columns and 'Lifetime' in df_engineered.columns:
        df_engineered['Avg_additional_charges_per_month'] = df_engineered['Avg_additional_charges_total'] / np.maximum(df_engineered['Lifetime'], 1)
    
    # Create Class_frequency_change if both Avg_class_frequency_current_month and Avg_class_frequency_total exist
    if 'Avg_class_frequency_current_month' in df_engineered.columns and 'Avg_class_frequency_total' in df_engineered.columns:
        df_engineered['Class_frequency_change'] = df_engineered['Avg_class_frequency_current_month'] - df_engineered['Avg_class_frequency_total']
    
    # Create Is_regular if Avg_class_frequency_total exists
    if 'Avg_class_frequency_total' in df_engineered.columns:
        df_engineered['Is_regular'] = df_engineered['Avg_class_frequency_total'] > df_engineered['Avg_class_frequency_total'].median()
    
    # Convert Contract_length to numeric if it exists
    if 'Contract_length' in df_engineered.columns:
        contract_length_map = {'Short': 0, 'Medium': 1, 'Long': 2}
        df_engineered['Contract_length_numeric'] = df_engineered['Contract_length'].map(contract_length_map)
        df_engineered = df_engineered.drop(['Contract_length'], axis=1)
    
    return df_engineered

def engineer_scenario_features(df, scenario_params):
    df = engineer_features(df)
    
    # Add scenario-specific feature engineering
    if 'new_classes' in scenario_params:
        df['class_variety_score'] = df['class_variety_score'] * (1 + scenario_params['new_classes'] / df['available_classes'])
    
    if 'facility_improvement' in scenario_params:
        df['facility_score'] = df['facility_score'] * (1 + scenario_params['facility_improvement'])
    
    if 'staff_training' in scenario_params:
        df['staff_satisfaction_score'] = df['staff_satisfaction_score'] * (1 + scenario_params['staff_training'])
    
    # Add more scenario-specific feature engineering as needed
    
    return df