from rest_framework import serializers
from django.core.mail import send_mail
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from .models import User, Institution
import json

class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'password')
        extra_kwargs = {
            'password': {'write_only': True}
        }

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        
        domain = value.split('@')[1]
        institutions = Institution.objects.all()
        
        allowed_institution = next((inst for inst in institutions if inst.is_domain_allowed(domain)), None)
        
        if not allowed_institution:
            raise serializers.ValidationError("This email domain is not allowed.")
        
        return value

    def validate_password(self, value):
        try:
            validate_password(value)
        except DjangoValidationError as e:
            raise serializers.ValidationError(str(e))
        return value

    def create(self, validated_data):
        user = User.objects.create_user(
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name'],
            email=validated_data['email'],
            password=validated_data['password']
        )
        user.is_active = False
        user.is_email_verified = False
        
        # Set the user's institution based on their email domain
        domain = user.email.split('@')[1]
        institution = Institution.objects.filter(allowed_domains__contains=domain).first()
        if institution:
            user.institution = institution
        
        user.save()

        self.send_activation_email(user)

        return user
    
    def send_activation_email(self, user):
        current_site = get_current_site(self.context['request'])
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = default_token_generator.make_token(user)
        activation_link = f"http://{current_site.domain}/activate/{uid}/{token}/"

        context = {
            'user': user,
            'activation_link': activation_link,
        }
        html_message = render_to_string('emails/activation_email_template.html', context)
        plain_message = strip_tags(html_message)

        send_mail(
            'Activate your GymWise account',
            plain_message,
            'noreply@gymwise.com',
            [user.email],
            html_message=html_message,
            fail_silently=False,
        )

class ActivateUserSerializer(serializers.Serializer):
    uid = serializers.CharField()
    token = serializers.CharField()

class ForgotPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()

class ResetPasswordSerializer(serializers.Serializer):
    password = serializers.CharField(write_only=True, required=True)
    confirm_password = serializers.CharField(write_only=True, required=True)

    def validate_password(self, value):
        try:
            validate_password(value)
        except DjangoValidationError as e:
            raise serializers.ValidationError(str(e))
        return value

    def validate(self, attrs):
        if attrs['password'] != attrs['confirm_password']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

class ManageRolesSerializer(serializers.Serializer):
    email = serializers.EmailField()
    role = serializers.ChoiceField(choices=User.ROLE_CHOICES)

class InstitutionSerializer(serializers.ModelSerializer):
    allowed_domains = serializers.ListField(child=serializers.CharField(), write_only=True)

    class Meta:
        model = Institution
        fields = ['id', 'name', 'schema_name', 'subdomain', 'domain', 'allowed_domains', 'is_active']

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        representation['allowed_domains'] = json.loads(instance.allowed_domains)
        return representation

    def to_internal_value(self, data):
        if 'allowed_domains' in data:
            data['allowed_domains'] = json.dumps(data['allowed_domains'])
        return super().to_internal_value(data)

class UserSerializer(serializers.ModelSerializer):
    institution_name = serializers.CharField(source='institution.name', read_only=True)

    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'role', 'institution', 'institution_name', 'is_active', 'is_email_verified']

class InactiveUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'institution']