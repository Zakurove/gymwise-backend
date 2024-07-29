from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os
from django.conf import settings
from ..models import InstitutionModel, Institution
import logging

logger = logging.getLogger(__name__)

def train_base_model(X, y):
    model = XGBClassifier(random_state=42)
    model.fit(X, y)
    return model

def fine_tune_model(base_model, X, y):
    fine_tuned_model = joblib.load(joblib.dumps(base_model))
    fine_tuned_model.fit(X, y, xgb_model=fine_tuned_model)
    return fine_tuned_model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    auc_roc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return accuracy, auc_roc

def train_models(X, y, institution_id=None):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if institution_id is None:
            # Train base model
            model = train_base_model(X_train, y_train)
            model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', 'base_model.joblib')
            joblib.dump(model, model_path)
            accuracy, auc_roc = evaluate_model(model, X_test, y_test)
            logger.info(f"Base model trained. Accuracy: {accuracy}, AUC-ROC: {auc_roc}")
            return model, accuracy, auc_roc
        else:
            # Fine-tune model for specific institution
            base_model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', 'base_model.joblib')
            base_model = joblib.load(base_model_path)
            fine_tuned_model = fine_tune_model(base_model, X_train, y_train)
            
            # Save fine-tuned model
            model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', f'institution_{institution_id}_model.joblib')
            joblib.dump(fine_tuned_model, model_path)
            
            accuracy, auc_roc = evaluate_model(fine_tuned_model, X_test, y_test)
            
            # Update or create InstitutionModel instance
            institution = Institution.objects.get(id=institution_id)
            InstitutionModel.objects.update_or_create(
                institution=institution,
                defaults={
                    'model_path': model_path,
                    'performance_metrics': {
                        'accuracy': accuracy,
                        'auc_roc': auc_roc
                    }
                }
            )
            
            logger.info(f"Fine-tuned model for institution {institution_id}. Accuracy: {accuracy}, AUC-ROC: {auc_roc}")
            return fine_tuned_model, accuracy, auc_roc
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        return None, None, None

def get_model_for_institution(institution_id):
    try:
        institution_model = InstitutionModel.objects.get(institution_id=institution_id)
        return joblib.load(institution_model.model_path)
    except InstitutionModel.DoesNotExist:
        # If no specific model exists, return the base model
        base_model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', 'base_model.joblib')
        return joblib.load(base_model_path)
    except Exception as e:
        logger.error(f"Error in get_model_for_institution: {str(e)}")
        return None