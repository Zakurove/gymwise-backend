from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from django.db import connection
from .ml import data_preprocessing, feature_engineering, model_training, model_evaluation, customer_segmentation
from .models import Member, ActionableInsight, ModelMetrics, FeatureImportance, MemberSegment, MemberSegmentAssignment, MemberActivity, InstitutionModel
from .insights import generate_insights
from .data_pipeline import get_preprocessed_data_for_institution
from django.db.models import Count
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class UploadMemberDataView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request, format=None):
        try:
            institution = request.user.institution
            if not institution:
                return Response({'error': 'User is not associated with an institution'}, status=400)

            file = request.FILES.get('file')
            if not file:
                return Response({'error': 'No file uploaded'}, status=400)
            
            df = pd.read_csv(file)
            df = data_preprocessing.preprocess_data(df)
            df = feature_engineering.engineer_features(df)
            
            X = df.drop(['Churn', 'name', 'email'], axis=1)
            y = df['Churn']
            
            # Train or fine-tune model for the specific institution
            model = model_training.train_models(X, y, institution.id)
            
            # Make predictions
            probabilities = model.predict_proba(X)[:, 1]
            
            df['churn_probability'] = probabilities
            df['churn_risk'] = pd.cut(df['churn_probability'], 
                                      bins=[0, 0.3, 0.7, 1], 
                                      labels=['low', 'medium', 'high'])
            
            # Update or create Member objects
            for _, row in df.iterrows():
                member, created = Member.objects.update_or_create(
                    email=row['email'],
                    institution=institution,
                    defaults={
                        'name': row['name'],
                        'churn_risk': row['churn_risk'],
                        'churn_probability': row['churn_probability'],
                        # Add other fields as necessary
                    }
                )
                
                # Generate insights
                historical_data = MemberActivity.objects.filter(member=member)
                insights = generate_insights(member, historical_data)
                for insight in insights:
                    ActionableInsight.objects.create(
                        member=member, 
                        type=insight['type'],
                        message=insight['message'],
                        institution=institution
                    )
            
            return Response({'message': 'Data processed successfully'})
        
        except Exception as e:
            logger.error(f"Error in UploadMemberDataView: {str(e)}", exc_info=True)
            return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_member_insights(request):
    try:
        member = Member.objects.get(user=request.user, institution=request.user.institution)
        historical_data = MemberActivity.objects.filter(member=member)
        insights = generate_insights(member, historical_data)
        actionable_insights = ActionableInsight.objects.filter(member=member, institution=request.user.institution).order_by('-created_at')
        
        response_data = {
            'member_info': {
                'name': member.name,
                'churn_risk': member.churn_risk,
                'churn_probability': member.churn_probability,
            },
            'insights': insights,
            'actionable_insights': [
                {
                    'type': insight.type,
                    'message': insight.message,
                    'created_at': insight.created_at
                } for insight in actionable_insights
            ]
        }
        
        return Response(response_data)
    except Member.DoesNotExist:
        return Response({'error': 'Member not found'}, status=404)
    except Exception as e:
        logger.error(f"Error in get_member_insights: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while fetching member insights'}, status=500)


@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_churn_risk_distribution(request):
    try:
        distribution = Member.objects.filter(institution=request.user.institution).values('churn_risk').annotate(count=Count('churn_risk'))
        return Response(distribution)
    except Exception as e:
        logger.error(f"Error in get_churn_risk_distribution: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while fetching churn risk distribution'}, status=500)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_model_performance(request):
    try:
        latest_metrics = ModelMetrics.objects.filter(institution=request.user.institution).latest('date')
        performance_data = {
            'accuracy': latest_metrics.accuracy,
            'precision': latest_metrics.precision,
            'recall': latest_metrics.recall,
            'f1_score': latest_metrics.f1_score,
            'auc_roc': latest_metrics.auc_roc,
            'date': latest_metrics.date
        }
        return Response(performance_data)
    except ModelMetrics.DoesNotExist:
        return Response({'error': 'No model performance data available'}, status=404)
    except Exception as e:
        logger.error(f"Error in get_model_performance: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while fetching model performance'}, status=500)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_feature_importance(request):
    try:
        feature_importance = FeatureImportance.objects.filter(institution=request.user.institution).order_by('-importance_score')[:10]
        data = [{'feature': fi.feature_name, 'importance': fi.importance_score} for fi in feature_importance]
        return Response(data)
    except Exception as e:
        logger.error(f"Error in get_feature_importance: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while fetching feature importance'}, status=500)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_member_segments(request):
    try:
        segments = MemberSegment.objects.filter(institution=request.user.institution).annotate(member_count=Count('membersegmentassignment'))
        data = [{'segment': segment.name, 'count': segment.member_count} for segment in segments]
        return Response(data)
    except Exception as e:
        logger.error(f"Error in get_member_segments: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while fetching member segments'}, status=500)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def retrain_model(request):
    try:
        # Load and preprocess data for the specific institution
        df = get_preprocessed_data_for_institution(request.user.institution.id)
        
        X = df.drop('churn_probability', axis=1)
        y = df['churn_probability'].apply(lambda x: 1 if x > 0.5 else 0)  # Convert to binary

        # Retrain the model
        model = model_training.train_models(X, y, request.user.institution.id)
        
        # Evaluate the new model
        results = model_evaluation.evaluate_model(model, X, y)
        
        # Save the model metrics
        ModelMetrics.objects.create(
            institution=request.user.institution,
            accuracy=results['accuracy'],
            precision=results['precision'],
            recall=results['recall'],
            f1_score=results['f1_score'],
            auc_roc=results['auc_roc']
        )
        
        return Response({'message': 'Model retrained successfully', 'performance': results})
    except Exception as e:
        logger.error(f"Error in retrain_model: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while retraining the model'}, status=500)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_engagement_metrics(request):
    try:
        total_members = Member.objects.filter(institution=request.user.institution).count()
        active_members = Member.objects.filter(institution=request.user.institution, visit_frequency__gt=2).count()
        engagement_rate = active_members / total_members if total_members > 0 else 0
        
        metrics = {
            'total_members': total_members,
            'active_members': active_members,
            'engagement_rate': engagement_rate
        }
        return Response(metrics)
    except Exception as e:
        logger.error(f"Error in get_engagement_metrics: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while fetching engagement metrics'}, status=500)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_retention_forecast(request):
    try:
        total_members = Member.objects.filter(institution=request.user.institution).count()
        at_risk_members = Member.objects.filter(institution=request.user.institution, churn_risk='high').count()
        forecast_retention_rate = 1 - (at_risk_members / total_members) if total_members > 0 else 0
        
        forecast = {
            'current_members': total_members,
            'at_risk_members': at_risk_members,
            'forecast_retention_rate': forecast_retention_rate
        }
        return Response(forecast)
    except Exception as e:
        logger.error(f"Error in get_retention_forecast: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while generating retention forecast'}, status=500)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_seasonal_insights(request):
    try:
        current_month = datetime.now().month
        
        if 5 <= current_month <= 8:  # Summer months
            insight = "Summer season: Focus on outdoor activities and hydration reminders."
        elif current_month == 9:  # Ramadan (approximate)
            insight = "Ramadan: Adjust class schedules and offer nutrition advice for fasting members."
        else:
            insight = "Regular season: Maintain standard engagement strategies."
        
        return Response({'seasonal_insight': insight})
    except Exception as e:
        logger.error(f"Error in get_seasonal_insights: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while generating seasonal insights'}, status=500)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def train_and_evaluate_model(request):
    try:
        institution = request.user.institution
        logger.info(f"Starting model training and evaluation for {institution.name}")
        
        # Load data for the specific institution
        df = get_preprocessed_data_for_institution(institution.id)
        logger.info("Data loaded and preprocessed")
        
        X = df.drop('churn_probability', axis=1)
        y = df['churn_probability'].apply(lambda x: 1 if x > 0.5 else 0)  # Convert to binary
        
        # Train model for the specific institution
        model = model_training.train_models(X, y, institution.id)
        logger.info("Model trained")
        
        # Evaluate model
        results = model_evaluation.evaluate_model(model, X, y)
        logger.info("Model evaluated")
        
        # Save model metrics
        ModelMetrics.objects.create(
            institution=institution,
            accuracy=results['accuracy'],
            precision=results['precision'],
            recall=results['recall'],
            f1_score=results['f1_score'],
            auc_roc=results['auc_roc']
        )
        
        # Save or update InstitutionModel
        model_path = f'models/institution_{institution.id}_model.joblib'
        joblib.dump(model, model_path)
        InstitutionModel.objects.update_or_create(
            institution=institution,
            defaults={
                'model_path': model_path,
                'performance_metrics': results
            }
        )
        
        return JsonResponse({
            'message': 'Model trained and evaluated successfully',
            'results': results
        })
    except Exception as e:
        logger.error(f"Error in train_and_evaluate_model: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)