# gymwise-backend/ai/tests.py

from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from accounts.models import User, Institution
from .models import Member, MemberActivity, ModelMetrics
from django_pgschemas.utils import schema_context
import json

class AIViewsTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.institution = Institution.objects.create(name="Test Gym", schema_name="test_gym")
        with schema_context(self.institution.schema_name):
            self.user = User.objects.create_user(username="testuser", password="testpass", institution=self.institution)
        self.client.force_authenticate(user=self.user)

    def test_get_member_insights(self):
        with schema_context(self.institution.schema_name):
            Member.objects.create(user=self.user, institution=self.institution, name="Test Member", email="test@example.com")
        response = self.client.get(reverse('member_insights'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('member_info', response.data)

    def test_get_churn_risk_distribution(self):
        response = self.client.get(reverse('churn_risk_distribution'))
        self.assertEqual(response.status_code, 200)

    def test_get_model_performance(self):
        with schema_context(self.institution.schema_name):
            ModelMetrics.objects.create(
                institution=self.institution,
                accuracy=0.9,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                auc_roc=0.85
            )
        response = self.client.get(reverse('model_performance'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('accuracy', response.data)

    def test_get_feature_importance(self):
        response = self.client.get(reverse('feature_importance'))
        self.assertEqual(response.status_code, 200)

    def test_get_member_segments(self):
        response = self.client.get(reverse('member_segments'))
        self.assertEqual(response.status_code, 200)

    def test_get_engagement_metrics(self):
        response = self.client.get(reverse('engagement_metrics'))
        self.assertEqual(response.status_code, 200)

    def test_get_retention_forecast(self):
        response = self.client.get(reverse('retention_forecast'))
        self.assertEqual(response.status_code, 200)

    def test_get_seasonal_insights(self):
        response = self.client.get(reverse('seasonal_insights'))
        self.assertEqual(response.status_code, 200)

class AIModelsTestCase(TestCase):
    def setUp(self):
        self.institution = Institution.objects.create(name="Test Gym", schema_name="test_gym")
        with schema_context(self.institution.schema_name):
            self.user = User.objects.create_user(username="testuser", password="testpass", institution=self.institution)

    def test_member_creation(self):
        with schema_context(self.institution.schema_name):
            member = Member.objects.create(user=self.user, institution=self.institution, name="Test Member", email="test@example.com")
            self.assertEqual(Member.objects.count(), 1)
            self.assertEqual(member.name, "Test Member")

    def test_member_activity_creation(self):
        with schema_context(self.institution.schema_name):
            member = Member.objects.create(user=self.user, institution=self.institution, name="Test Member", email="test@example.com")
            activity = MemberActivity.objects.create(member=member, date="2023-01-01", class_name="Test Class", time_of_day="morning")
            self.assertEqual(MemberActivity.objects.count(), 1)
            self.assertEqual(activity.class_name, "Test Class")