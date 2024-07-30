from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
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

def train_models(X, y, institution_id=None):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if institution_id is None:
            model = train_base_model(X_train, y_train)
            model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', 'base_model.joblib')
        else:
            base_model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', 'base_model.joblib')
            base_model = joblib.load(base_model_path)
            model = fine_tune_model(base_model, X_train, y_train)
            model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', f'institution_{institution_id}_model.joblib')

        joblib.dump(model, model_path)

        if institution_id is not None:
            institution = Institution.objects.get(id=institution_id)
            InstitutionModel.objects.update_or_create(
                institution=institution,
                defaults={'model_path': model_path}
            )

        logger.info(f"Model trained and saved for institution {institution_id}")
        return model

    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        return None

def get_model_for_institution(institution_id):
    try:
        institution_model = InstitutionModel.objects.get(institution_id=institution_id)
        return joblib.load(institution_model.model_path)
    except InstitutionModel.DoesNotExist:
        base_model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', 'base_model.joblib')
        return joblib.load(base_model_path)
    except Exception as e:
        logger.error(f"Error in get_model_for_institution: {str(e)}")
        return None

def predict_scenario(model, X_scenario):
    try:
        return model.predict_proba(X_scenario)[:, 1]
    except Exception as e:
        logger.error(f"Error in predict_scenario: {str(e)}")
        return None