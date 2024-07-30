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
from .models import Campaign, CampaignPerformance
from rest_framework import status
from django.db.models import Sum, Avg

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
    




class CampaignSuggestionsView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        try:
            institution_id = request.user.institution.id
            
            # Load member data
            members = Member.objects.filter(institution_id=institution_id).values()
            df = pd.DataFrame.from_records(members)
            
            # Preprocess data
            df = data_preprocessing.preprocess_data(df)
            df = feature_engineering.engineer_features(df)
            
            # Get model
            model = model_training.get_model_for_institution(institution_id)
            
            if model is None:
                return Response({"error": "Model not found for this institution"}, status=status.HTTP_404_NOT_FOUND)
            
            # Predict churn probabilities
            X = df.drop(['id', 'name', 'email', 'churn_probability'], axis=1, errors='ignore')
            churn_probabilities = model.predict_proba(X)[:, 1]
            
            # Segment members based on churn risk
            df['churn_risk'] = pd.cut(churn_probabilities, bins=[0, 0.3, 0.7, 1], labels=['Low', 'Medium', 'High'])
            
            # Generate campaign suggestions
            suggestions = self.generate_campaign_suggestions(df)
            
            return Response(suggestions, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Error in CampaignSuggestionsView: {str(e)}")
            return Response({"error": "An error occurred while generating campaign suggestions"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def generate_campaign_suggestions(self, df):
        suggestions = []
        
        # High-risk members campaign
        high_risk_count = df[df['churn_risk'] == 'High'].shape[0]
        if high_risk_count > 0:
            suggestions.append({
                "target_segment": "High Churn Risk",
                "member_count": high_risk_count,
                "campaign_type": "Personalized Retention",
                "message_template": "We've missed you at the gym! Here's a special offer to get you back on track: {offer}",
                "recommended_actions": ["Offer a free personal training session", "Provide a discount on membership renewal"]
            })
        
        # Low-engagement members campaign
        low_engagement_count = df[df['visit_frequency'] < df['visit_frequency'].median()].shape[0]
        if low_engagement_count > 0:
            suggestions.append({
                "target_segment": "Low Engagement",
                "member_count": low_engagement_count,
                "campaign_type": "Re-engagement",
                "message_template": "Boost your fitness journey with our new {class_type} class! Join us this week and feel the difference.",
                "recommended_actions": ["Introduce new class types", "Send workout tips and motivation"]
            })
        
        # Expiring membership campaign
        expiring_soon_count = df[df['days_to_membership_expiry'] <= 30].shape[0]
        if expiring_soon_count > 0:
            suggestions.append({
                "target_segment": "Expiring Memberships",
                "member_count": expiring_soon_count,
                "campaign_type": "Renewal",
                "message_template": "Your membership is expiring soon. Renew now and get {discount}% off your next month!",
                "recommended_actions": ["Offer renewal incentives", "Highlight new gym features or classes"]
            })
        
        return suggestions

class CreateCampaignView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        try:
            campaign_data = request.data
            campaign_data['institution'] = request.user.institution
            
            campaign = Campaign.objects.create(**campaign_data)
            
            return Response({"message": "Campaign created successfully", "id": campaign.id}, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            logger.error(f"Error in CreateCampaignView: {str(e)}")
            return Response({"error": "An error occurred while creating the campaign"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CampaignPerformanceView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        try:
            institution_id = request.user.institution.id
            campaigns = Campaign.objects.filter(institution_id=institution_id)
            
            performance_data = []
            for campaign in campaigns:
                performance = CampaignPerformance.objects.filter(campaign=campaign).aggregate(
                    avg_open_rate=Avg('open_rate'),
                    avg_click_rate=Avg('click_through_rate'),
                    avg_conversion_rate=Avg('conversion_rate')
                )
                
                performance_data.append({
                    "id": campaign.id,
                    "name": campaign.name,
                    "target_segment": campaign.target_segment,
                    "campaign_type": campaign.campaign_type,
                    "start_date": campaign.start_date,
                    "end_date": campaign.end_date,
                    "status": campaign.status,
                    "avg_open_rate": performance['avg_open_rate'],
                    "avg_click_rate": performance['avg_click_rate'],
                    "avg_conversion_rate": performance['avg_conversion_rate']
                })
            
            return Response(performance_data, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Error in CampaignPerformanceView: {str(e)}")
            return Response({"error": "An error occurred while fetching campaign performance"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_active_campaigns(request):
    try:
        campaigns = Campaign.objects.filter(institution=request.user.institution, status='Active')
        data = [{
            'id': campaign.id,
            'name': campaign.name,
            'targetSegment': campaign.target_segment,
            'campaignType': campaign.campaign_type,
            'status': campaign.status,
            'startDate': campaign.start_date,
            'endDate': campaign.end_date,
        } for campaign in campaigns]
        return Response(data)
    except Exception as e:
        logger.error(f"Error in get_active_campaigns: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while fetching active campaigns'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def get_campaign_insights(request):
    try:
        total_campaigns = Campaign.objects.filter(institution=request.user.institution).count()
        active_campaigns = Campaign.objects.filter(institution=request.user.institution, status='Active').count()
        
        total_sent = CampaignPerformance.objects.filter(campaign__institution=request.user.institution).aggregate(Sum('total_sent'))['total_sent__sum'] or 0
        avg_open_rate = CampaignPerformance.objects.filter(campaign__institution=request.user.institution).aggregate(Avg('open_rate'))['open_rate__avg'] or 0
        avg_ctr = CampaignPerformance.objects.filter(campaign__institution=request.user.institution).aggregate(Avg('click_through_rate'))['click_through_rate__avg'] or 0
        avg_conversion_rate = CampaignPerformance.objects.filter(campaign__institution=request.user.institution).aggregate(Avg('conversion_rate'))['conversion_rate__avg'] or 0

        insights = [
            {'label': 'Total Campaigns', 'value': str(total_campaigns), 'change': 0},
            {'label': 'Active Campaigns', 'value': str(active_campaigns), 'change': 0},
            {'label': 'Total Sent', 'value': str(total_sent), 'change': 0},
            {'label': 'Avg. Open Rate', 'value': f"{avg_open_rate:.2f}%", 'change': 0},
            {'label': 'Avg. Click-through Rate', 'value': f"{avg_ctr:.2f}%", 'change': 0},
            {'label': 'Avg. Conversion Rate', 'value': f"{avg_conversion_rate:.2f}%", 'change': 0},
        ]
        return Response(insights)
    except Exception as e:
        logger.error(f"Error in get_campaign_insights: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while fetching campaign insights'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class WhatIfScenarioView(APIView):
    def post(self, request):
        try:
            institution_id = request.user.institution.id
            scenario_params = request.data.get('scenario_params', {})

            # Input validation
            if not self.validate_scenario_params(scenario_params):
                return Response({"error": "Invalid scenario parameters"}, status=status.HTTP_400_BAD_REQUEST)

            # Check cache for existing scenario results
            cache_key = f"scenario_{institution_id}_{hash(frozenset(scenario_params.items()))}"
            cached_result = cache.get(cache_key)
            if cached_result:
                return Response(cached_result, status=status.HTTP_200_OK)

            # Load current member data
            members = Member.objects.filter(institution_id=institution_id).values()
            df = pd.DataFrame.from_records(members)

            # Preprocess and engineer features for the scenario
            df = data_preprocessing.preprocess_scenario_data(df, scenario_params)
            df = feature_engineering.engineer_scenario_features(df, scenario_params)

            # Get the model for this institution
            model = model_training.get_model_for_institution(institution_id)

            if model is None:
                return Response({"error": "Model not found for this institution"}, status=status.HTTP_404_NOT_FOUND)

            # Make predictions
            X_scenario = df.drop(['id', 'name', 'email', 'churn_probability'], axis=1, errors='ignore')
            scenario_probabilities = model_training.predict_scenario(model, X_scenario)

            if scenario_probabilities is None:
                return Response({"error": "Error in scenario prediction"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Calculate retention rate
            current_retention_rate = 1 - df['churn_probability'].mean()
            scenario_retention_rate = 1 - scenario_probabilities.mean()

            response_data = {
                "current_retention_rate": current_retention_rate,
                "scenario_retention_rate": scenario_retention_rate,
                "retention_rate_change": scenario_retention_rate - current_retention_rate,
            }

            # Cache the result
            cache.set(cache_key, response_data, timeout=3600)  # Cache for 1 hour

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in WhatIfScenarioView: {str(e)}")
            return Response({"error": "An error occurred while processing the scenario"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def validate_scenario_params(self, params):
        valid_params = {
            'membership_price_change': (-0.5, 0.5),
            'new_classes': (0, 10),
            'gym_hours_change': (-4, 4),
            'marketing_intensity': (0.5, 2),
            'facility_improvement': (0, 0.5),
            'staff_training': (0, 0.5)
        }

        for param, (min_val, max_val) in valid_params.items():
            if param in params:
                if not isinstance(params[param], (int, float)) or params[param] < min_val or params[param] > max_val:
                    return False
        return True
