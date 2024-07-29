from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'classification_report': report,
        'roc_auc_score': auc_score,
        'confusion_matrix': cm.tolist()
    }

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        return None

    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return feature_importance