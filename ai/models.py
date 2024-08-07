from django.db import models
from django.contrib.auth import get_user_model
from accounts.models import Institution, TenantAwareModel
import json

User = get_user_model()

class Member(TenantAwareModel):
    CHURN_RISK_CHOICES = [
        ('low', 'Low Risk'),
        ('medium', 'Medium Risk'),
        ('high', 'High Risk'),
    ]

    institution = models.ForeignKey(Institution, on_delete=models.CASCADE, related_name='ai_members')
    email = models.EmailField()
    name = models.CharField(max_length=255)
    churn_risk = models.CharField(max_length=10, choices=CHURN_RISK_CHOICES, default='low')
    churn_probability = models.FloatField(default=0.0)
    last_prediction_date = models.DateTimeField(auto_now=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    membership_duration = models.IntegerField(null=True, blank=True)
    visit_frequency = models.FloatField(null=True, blank=True)
    join_date = models.DateField(null=True, blank=True)
    extended_data = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = ['email', 'institution']

    def __str__(self):
        return f"{self.name} - {self.institution.name} - {self.churn_risk} Risk"

class MappingTemplate(TenantAwareModel):
    name = models.CharField(max_length=100)
    mapping = models.JSONField()
    is_default = models.BooleanField(default=False)

    class Meta:
        unique_together = ('name', 'institution')

    def __str__(self):
        return f"{self.name} - {self.institution.name}"

class MemberActivity(TenantAwareModel):
    TIME_CHOICES = [
        ('morning', 'Morning'),
        ('afternoon', 'Afternoon'),
        ('evening', 'Evening'),
    ]
    member = models.ForeignKey(Member, on_delete=models.CASCADE, related_name='activities')
    date = models.DateField()
    class_name = models.CharField(max_length=255)
    time_of_day = models.CharField(max_length=10, choices=TIME_CHOICES)

    def __str__(self):
        return f"{self.member.name} - {self.class_name} on {self.date}"

class ActionableInsight(TenantAwareModel):
    member = models.ForeignKey(Member, on_delete=models.CASCADE, related_name='insights')
    type = models.CharField(max_length=50)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.type} insight for {self.member.name}"
    
class ModelMetrics(TenantAwareModel):
    date = models.DateTimeField(auto_now_add=True)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    auc_roc = models.FloatField()

    def __str__(self):
        return f"Metrics for {self.institution.name} on {self.date}"

class FeatureImportance(TenantAwareModel):
    feature_name = models.CharField(max_length=255)
    importance_score = models.FloatField()
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.feature_name}: {self.importance_score} for {self.institution.name}"

class MemberSegment(TenantAwareModel):
    name = models.CharField(max_length=255)
    description = models.TextField()

    def __str__(self):
        return f"{self.name} for {self.institution.name}"

class MemberSegmentAssignment(TenantAwareModel):
    member = models.ForeignKey(Member, on_delete=models.CASCADE)
    segment = models.ForeignKey(MemberSegment, on_delete=models.CASCADE)
    assignment_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('member', 'segment')

    def __str__(self):
        return f"{self.member.name} - {self.segment.name}"

class InstitutionModel(TenantAwareModel):
    model_path = models.CharField(max_length=255)
    last_trained = models.DateTimeField(auto_now=True)
    performance_metrics = models.JSONField(default=dict)

    def __str__(self):
        return f"AI Model for {self.institution.name}"

    def get_performance_metrics(self):
        return json.loads(self.performance_metrics)

    def set_performance_metrics(self, metrics):
        self.performance_metrics = json.dumps(metrics)



class Campaign(TenantAwareModel):
    name = models.CharField(max_length=255)
    target_segment = models.CharField(max_length=50)
    campaign_type = models.CharField(max_length=50)
    message = models.TextField()
    start_date = models.DateField()
    end_date = models.DateField()
    ai_optimization = models.BooleanField(default=False)
    status = models.CharField(max_length=20, default='Active')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.institution.name}"

class CampaignPerformance(TenantAwareModel):
    campaign = models.ForeignKey(Campaign, on_delete=models.CASCADE, related_name='performances')
    date = models.DateField()
    total_sent = models.IntegerField()
    open_rate = models.FloatField()
    click_through_rate = models.FloatField()
    conversion_rate = models.FloatField()

    def __str__(self):
        return f"Performance for {self.campaign.name} on {self.date}"