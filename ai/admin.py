from django.contrib import admin
from .models import Member, ActionableInsight, ModelMetrics, FeatureImportance, MemberSegment, MemberSegmentAssignment, MemberActivity, InstitutionModel, MappingTemplate, Campaign, CampaignPerformance

@admin.register(Member)
class MemberAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'institution', 'churn_risk', 'churn_probability')
    list_filter = ('institution', 'churn_risk')
    search_fields = ('name', 'email')

@admin.register(ActionableInsight)
class ActionableInsightAdmin(admin.ModelAdmin):
    list_display = ('member', 'type', 'created_at')
    list_filter = ('type', 'created_at')

@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = ('institution', 'date', 'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc')

@admin.register(FeatureImportance)
class FeatureImportanceAdmin(admin.ModelAdmin):
    list_display = ('institution', 'feature_name', 'importance_score', 'date')

@admin.register(MemberSegment)
class MemberSegmentAdmin(admin.ModelAdmin):
    list_display = ('name', 'institution')

@admin.register(MemberSegmentAssignment)
class MemberSegmentAssignmentAdmin(admin.ModelAdmin):
    list_display = ('member', 'segment', 'assignment_date')

@admin.register(MemberActivity)
class MemberActivityAdmin(admin.ModelAdmin):
    list_display = ('member', 'date', 'class_name', 'time_of_day')

@admin.register(InstitutionModel)
class InstitutionModelAdmin(admin.ModelAdmin):
    list_display = ('institution', 'model_path', 'last_trained')

@admin.register(MappingTemplate)
class MappingTemplateAdmin(admin.ModelAdmin):
    list_display = ('name', 'institution', 'is_default')

@admin.register(Campaign)
class CampaignAdmin(admin.ModelAdmin):
    list_display = ('name', 'institution', 'target_segment', 'start_date', 'end_date', 'status')

@admin.register(CampaignPerformance)
class CampaignPerformanceAdmin(admin.ModelAdmin):
    list_display = ('campaign', 'date', 'total_sent', 'open_rate', 'click_through_rate', 'conversion_rate')