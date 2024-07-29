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