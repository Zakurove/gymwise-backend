import joblib
import os
from django.conf import settings
from ..models import InstitutionModel, Institution
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import logging
from ..models import InstitutionModel, Institution
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

def create_flexible_pipeline(feature_names):
    numeric_features = feature_names
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
        ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    return model

def train_base_model(X, y):
    feature_names = X.columns.tolist()
    model = create_flexible_pipeline(feature_names)
    model.fit(X, y)
    return model, feature_names

def fine_tune_model(base_model, X, y):
    feature_names = X.columns.tolist()
    model = create_flexible_pipeline(feature_names)
    model.set_params(**base_model.get_params())
    model.fit(X, y)
    return model, feature_names

def train_institution_model(X, y, institution_id):
    try:
        base_model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', 'best_model.joblib')
        
        if not os.path.exists(base_model_path):
            logger.error("Base model not found. Please ensure the base model is trained and saved.")
            return None, None

        base_model = joblib.load(base_model_path)
        
        model, feature_names = fine_tune_model(base_model, X, y)
        model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', f'institution_{institution_id}_model.joblib')
        joblib.dump((model, feature_names), model_path)

        institution = Institution.objects.get(id=institution_id)
        InstitutionModel.objects.update_or_create(
            institution=institution,
            defaults={'model_path': model_path}
        )

        logger.info(f"Model trained and saved for institution {institution_id}")
        return model, feature_names

    except Exception as e:
        logger.error(f"Error in train_institution_model: {str(e)}", exc_info=True)
        return None, None

def get_model_for_institution(institution_id):
    try:
        institution_model = InstitutionModel.objects.get(institution_id=institution_id)
        loaded_object = joblib.load(institution_model.model_path)
        if isinstance(loaded_object, tuple) and len(loaded_object) == 2:
            model, feature_names = loaded_object
        else:
            model = loaded_object
            feature_names = None
        return model, feature_names
    except InstitutionModel.DoesNotExist:
        logger.warning(f"No specific model found for institution {institution_id}. Using base model.")
        base_model_path = os.path.join(settings.BASE_DIR, 'ai', 'models', 'best_model.joblib')
        model = joblib.load(base_model_path)
        # We'll return None for feature_names, and handle this in the view
        return model, None
    except Exception as e:
        logger.error(f"Error in get_model_for_institution: {str(e)}", exc_info=True)
        return None, None
    
def predict_scenario(model, X_scenario):
    try:
        return model.predict_proba(X_scenario)[:, 1]
    except Exception as e:
        logger.error(f"Error in predict_scenario: {str(e)}")
        return None