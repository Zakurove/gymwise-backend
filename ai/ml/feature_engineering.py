import numpy as np

def engineer_features(df):
    df['Contract_length'] = df['Contract_period'].map({1: 'Short', 6: 'Medium', 12: 'Long'})
    df['Time_left_ratio'] = df['Month_to_end_contract'] / df['Contract_period']
    df['Avg_additional_charges_per_month'] = df['Avg_additional_charges_total'] / np.maximum(df['Lifetime'], 1)
    df['Class_frequency_change'] = df['Avg_class_frequency_current_month'] - df['Avg_class_frequency_total']
    df['Is_regular'] = df['Avg_class_frequency_total'] > df['Avg_class_frequency_total'].median()
    
    contract_length_map = {'Short': 0, 'Medium': 1, 'Long': 2}
    df['Contract_length_numeric'] = df['Contract_length'].map(contract_length_map)
    
    df = df.drop(['Contract_length'], axis=1)
    
    return df

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