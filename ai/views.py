from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from django.db import connection, transaction
from .ml import data_preprocessing, feature_engineering, model_training, model_evaluation, customer_segmentation
from .models import Member, ActionableInsight, ModelMetrics, FeatureImportance, MemberSegment, MemberSegmentAssignment, MemberActivity, InstitutionModel, MappingTemplate
from .insights import generate_insights
from .data_pipeline import get_preprocessed_data_for_institution
from django.db.models import Count
from datetime import datetime, timedelta
from accounts.permissions import IsSuperAdminOrAdmin
import pandas as pd
import numpy as np
import joblib
import os
from django.conf import settings
import logging
from .models import Campaign, CampaignPerformance
from rest_framework import status
from django.db.models import Sum, Avg
import json
from django.utils import timezone
from django.shortcuts import get_object_or_404
from django.core.paginator import Paginator, EmptyPage
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

        
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_member_insights(request):
    try:
        page = int(request.query_params.get('page', 1))
        sort_by = request.query_params.get('sort_by', 'name')
        sort_order = request.query_params.get('sort_order', 'asc')
        search = request.query_params.get('search', '')

        members_query = Member.objects.filter(institution=request.user.institution)
        
        if search:
            members_query = members_query.filter(
                Q(name__icontains=search) | Q(email__icontains=search)
            )
        
        sort_field = f"{'-' if sort_order == 'desc' else ''}{sort_by}"
        members = members_query.order_by(sort_field)

        members_data = []
        for member in members:
            historical_data = MemberActivity.objects.filter(member=member)
            insights = generate_insights(member, historical_data)
            actionable_insights = ActionableInsight.objects.filter(
                member=member, 
                institution=request.user.institution
            ).order_by('-created_at')
            
            member_data = {
                'member_info': {
                    'id': member.id,
                    'name': member.name,
                    'email': member.email,
                    'churn_risk': member.churn_risk,
                    'churn_probability': member.churn_probability,
                },
                'insights': insights,
                'actionable_insights': [
                    {
                        'type': insight.type,
                        'message': insight.message,
                        'created_at': insight.created_at.isoformat()
                    } for insight in actionable_insights
                ]
            }
            members_data.append(member_data)

        paginator = Paginator(members_data, 10)
        try:
            page_data = paginator.page(page)
        except EmptyPage:
            page_data = paginator.page(paginator.num_pages)

        response_data = {
            'results': page_data.object_list,
            'count': paginator.count,
            'page': page,
            'total_pages': paginator.num_pages
        }
        
        return Response(response_data)
    except Exception as e:
        logger.error(f"Error in get_member_insights: {str(e)}", exc_info=True)
        return Response({'error': 'An error occurred while fetching member insights'}, status=500)

def get_member_detail(request, member_id):
    member = get_object_or_404(Member, id=member_id)
    historical_data = MemberActivity.objects.filter(member=member)
    insights = generate_insights(member, historical_data)
    
    member_data = {
        'id': member.id,
        'name': member.name,
        'email': member.email,
        'churn_risk': member.churn_risk,
        'churn_probability': member.churn_probability,
        'visit_frequency': member.visit_frequency,
        'membership_duration': member.membership_duration,
        'join_date': member.join_date.isoformat() if member.join_date else None,
    }
    
    return JsonResponse({
        'member_info': member_data,
        'insights': insights,
        'actionable_insights': [
            {
                'type': insight.type,
                'message': insight.message,
                'created_at': insight.created_at.isoformat()
            } for insight in ActionableInsight.objects.filter(member=member).order_by('-created_at')
        ]
    })


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

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_data_status(request):
    institution = request.user.institution
    total_records = Member.objects.filter(institution=institution).count()
    processed_records = total_records  # Assume all records in Member are processed
    
    latest_metrics = ModelMetrics.objects.filter(institution=institution).order_by('-date').first()
    data_quality = latest_metrics.accuracy * 100 if latest_metrics else 0
    
    last_update = Member.objects.filter(institution=institution).order_by('-last_prediction_date').first()
    last_update_time = last_update.last_prediction_date if last_update else None

    return Response({
        'lastUpdate': last_update_time,
        'totalRecords': total_records,
        'processedRecords': processed_records,
        'dataQuality': data_quality
    })

class AnalyzeCSVView(APIView):
    permission_classes = [IsAuthenticated, IsSuperAdminOrAdmin]

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            df = pd.read_csv(file)
            columns = df.columns.tolist()
            return Response({'columns': columns})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class SaveMappingTemplateView(APIView):
    permission_classes = [IsAuthenticated, IsSuperAdminOrAdmin]

    def post(self, request):
        name = request.data.get('name')
        mapping = request.data.get('mapping')
        is_default = request.data.get('is_default', False)
        
        if not name or not mapping:
            return Response({'error': 'Name and mapping are required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            template, created = MappingTemplate.objects.update_or_create(
                name=name,
                institution=request.user.institution,
                defaults={
                    'mapping': mapping,
                    'is_default': is_default
                }
            )
            return Response({
                'id': template.id,
                'name': template.name,
                'is_default': template.is_default
            }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class GetMappingTemplatesView(APIView):
    permission_classes = [IsAuthenticated, IsSuperAdminOrAdmin]

    def get(self, request):
        templates = MappingTemplate.objects.filter(institution=request.user.institution)
        data = [{
            'id': t.id,
            'name': t.name,
            'is_default': t.is_default,
            'mapping': t.mapping
        } for t in templates]
        return Response(data)


class ProcessMappedDataView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file = request.FILES.get('file')
        mapping = request.data.get('mapping')
        
        if not file or not mapping:
            return Response({'error': 'File and mapping are required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            df = pd.read_csv(file)
            
            # Convert mapping from JSON string to dictionary if necessary
            if isinstance(mapping, str):
                mapping = json.loads(mapping)
            
            # Create a reverse mapping
            reverse_mapping = {v: k for k, v in mapping.items()}
            
            # Rename columns based on the mapping
            df_mapped = df.rename(columns=reverse_mapping)
            
            # Preprocess data
            df_processed = data_preprocessing.preprocess_data(df_mapped)
            df_processed = feature_engineering.engineer_features(df_processed)
            
            # Get the model and its features for this institution
            model, model_features = model_training.get_model_for_institution(request.user.institution.id)
            
            if model is None:
                logger.error("Failed to load the model.")
                return Response({'error': 'Failed to process data due to model loading error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Prepare data for prediction
            if model_features is not None:
                # Use only the features that are present in both model_features and df_processed
                common_features = list(set(model_features) & set(df_processed.columns))
                X = df_processed[common_features]
                logger.info(f"Using common features for prediction: {common_features}")
            else:
                # If model_features is None, use all numeric columns
                X = df_processed.select_dtypes(include=[np.number])
                logger.info(f"Using all numeric features for prediction: {X.columns.tolist()}")
            
            # Handle feature mismatch
            if X.shape[1] != model.n_features_in_:
                logger.warning(f"Feature mismatch. Model expects {model.n_features_in_} features, but got {X.shape[1]}. Adapting...")
                
                if X.shape[1] > model.n_features_in_:
                    # If we have more features than the model expects, select the first n_features_in_
                    X = X.iloc[:, :model.n_features_in_]
                else:
                    # If we have fewer features, add dummy columns
                    for i in range(X.shape[1], model.n_features_in_):
                        X[f'dummy_{i}'] = 0
                
                logger.info(f"Adapted features for prediction: {X.columns.tolist()}")
            
            # Ensure all columns are numeric
            X = X.apply(pd.to_numeric, errors='coerce')
            
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            
            # Scale features
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            # Make predictions
            probabilities = model.predict_proba(X)[:, 1]
            
            df_processed['churn_probability'] = probabilities
            df_processed['churn_risk'] = pd.cut(df_processed['churn_probability'], 
                                                bins=[0, 0.3, 0.7, 1], 
                                                labels=['low', 'medium', 'high'])

            
            with transaction.atomic():
                for _, row in df_processed.iterrows():
                    core_data = {
                        'email': row.get('email', f"user_{_}@example.com"),  # Use a default if email is missing
                        'name': row.get('name', f"User {_}"),  # Use a default if name is missing
                        'churn_risk': row['churn_risk'],
                        'churn_probability': float(row['churn_probability']),
                        'gender': row.get('gender'),
                        'age': float(row['Age']) if pd.notnull(row.get('Age')) else None,
                        'membership_duration': float(row['Lifetime']) if pd.notnull(row.get('Lifetime')) else None,
                        'visit_frequency': float(row['Avg_class_frequency_total']) if pd.notnull(row.get('Avg_class_frequency_total')) else None,
                    }
                    
                    # Handle extended data, replacing NaN with None
                    extended_data = {}
                    for col, value in row.items():
                        if col not in core_data:
                            if pd.isna(value):
                                extended_data[col] = None
                            elif isinstance(value, (int, float)):
                                extended_data[col] = float(value)
                            else:
                                extended_data[col] = str(value)

                    member, created = Member.objects.update_or_create(
                        email=core_data['email'],
                        institution=request.user.institution,
                        defaults={
                            **core_data,
                            'extended_data': extended_data,
                            'last_prediction_date': timezone.now()
                        }
                    )
                    
                    # Create MemberActivity record
                    MemberActivity.objects.create(
                        member=member,
                        institution=request.user.institution,
                        date=row.get('date', timezone.now().date()),
                        class_name=row.get('class_name', 'General'),
                        time_of_day=row.get('time_of_day', 'morning')
                    )
                    
                    # Generate insights
                    historical_data = MemberActivity.objects.filter(member=member)
                    insights = generate_insights(member, historical_data)
                    for insight in insights:
                        ActionableInsight.objects.create(
                            member=member, 
                            type=insight['type'],
                            message=insight['message'],
                            institution=request.user.institution
                        )
                    
                    logger.info(f"{'Created' if created else 'Updated'} member {member.id} with churn risk {member.churn_risk}")
            
            return Response({'message': 'Data processed and stored successfully'})
        except Exception as e:
            logger.error(f"Error in ProcessMappedDataView: {str(e)}", exc_info=True)
            return Response({'error': f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TrainInstitutionModelView(APIView):
    permission_classes = [IsAuthenticated, IsAdminUser]

    def post(self, request):
        file = request.FILES.get('file')
        mapping = request.data.get('mapping')
        
        if not file or not mapping:
            return Response({'error': 'Training data file and mapping are required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            df = pd.read_csv(file)
            
            # Apply mapping
            if isinstance(mapping, str):
                mapping = json.loads(mapping)
            reverse_mapping = {v: k for k, v in mapping.items()}
            df_mapped = df.rename(columns=reverse_mapping)
            
            # Preprocess data
            df_processed = data_preprocessing.preprocess_data(df_mapped)
            df_processed = feature_engineering.engineer_features(df_processed)
            
            # Prepare data for model
            X = df_processed.drop(['Churn', 'name', 'email'], axis=1, errors='ignore')
            y = df_processed['Churn']
            
            # Train model for the specific institution
            model, feature_names = model_training.train_institution_model(X, y, request.user.institution.id)
            
            if model is None:
                return Response({'error': 'Failed to train the model'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Save the mapping template
            MappingTemplate.objects.update_or_create(
                name=f"Institution_{request.user.institution.id}_Template",
                institution=request.user.institution,
                defaults={'mapping': mapping}
            )
            
            return Response({
                'message': 'Institution model trained successfully',
                'features_used': feature_names
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Error in TrainInstitutionModelView: {str(e)}", exc_info=True)
            return Response({'error': f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)