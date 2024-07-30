# gymwise-backend/ai/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('upload-member-data/', views.UploadMemberDataView.as_view(), name='upload_member_data'),
    path('member-insights/', views.get_member_insights, name='member_insights'),
    path('churn-risk-distribution/', views.get_churn_risk_distribution, name='churn_risk_distribution'),
    path('model-performance/', views.get_model_performance, name='model_performance'),
    path('feature-importance/', views.get_feature_importance, name='feature_importance'),
    path('member-segments/', views.get_member_segments, name='member_segments'),
    path('retrain-model/', views.retrain_model, name='retrain_model'),
    path('engagement-metrics/', views.get_engagement_metrics, name='engagement_metrics'),
    path('retention-forecast/', views.get_retention_forecast, name='retention_forecast'),
    path('seasonal-insights/', views.get_seasonal_insights, name='seasonal_insights'),
    path('train-and-evaluate/', views.train_and_evaluate_model, name='train_and_evaluate'),
    path('campaign-suggestions/', views.CampaignSuggestionsView.as_view(), name='campaign_suggestions'),
    path('create-campaign/', views.CreateCampaignView.as_view(), name='create_campaign'),
    path('campaign-performance/', views.CampaignPerformanceView.as_view(), name='campaign_performance'),
    path('active-campaigns/', views.get_active_campaigns, name='active_campaigns'),
    path('campaign-insights/', views.get_campaign_insights, name='campaign_insights'),
    path('what-if-scenario/', views.WhatIfScenarioView.as_view(), name='what_if_scenario'),
]