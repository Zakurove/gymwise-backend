from rest_framework import generics, status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_str
from django.contrib.auth.tokens import default_token_generator
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from .models import Institution
from .serializers import (
    RegisterSerializer, 
    ActivateUserSerializer, 
    ForgotPasswordSerializer, 
    ResetPasswordSerializer, 
    ManageRolesSerializer, 
    InstitutionSerializer,
    UserSerializer
)

User = get_user_model()

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]

    def perform_create(self, serializer):
        user = serializer.save()
        email_domain = user.email.split('@')[1]
        institution = Institution.objects.filter(allowed_domains__contains=email_domain).first()
        if institution:
            user.institution = institution
            user.save()

class ActivateUserView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, uidb64, token):
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            user.is_email_verified = True
            user.save()
            return Response({"detail": "Email verified successfully."}, status=status.HTTP_200_OK)
        else:
            return Response({"detail": "Invalid activation link."}, status=status.HTTP_400_BAD_REQUEST)

class CustomTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        if response.status_code == status.HTTP_200_OK:
            user = User.objects.get(email=request.data['email'])
            response.data['user'] = UserSerializer(user).data
        return response

class ForgotPasswordView(APIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = ForgotPasswordSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data['email']
        try:
            user = User.objects.get(email=email)
            # Generate password reset token
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            reset_link = f"{settings.FRONTEND_URL}/reset-password/{uid}/{token}"
            
            # Send email
            context = {'reset_link': reset_link, 'user': user}
            html_message = render_to_string('emails/password_reset_email_template.html', context)
            plain_message = strip_tags(html_message)
            send_mail(
                'Reset your password',
                plain_message,
                settings.DEFAULT_FROM_EMAIL,
                [email],
                html_message=html_message,
            )
            return Response({"detail": "Password reset email sent."}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({"detail": "User with this email does not exist."}, status=status.HTTP_404_NOT_FOUND)

class ResetPasswordView(APIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = ResetPasswordSerializer

    def post(self, request, uidb64, token):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            user.set_password(serializer.validated_data['password'])
            user.save()
            return Response({"detail": "Password has been reset successfully."}, status=status.HTTP_200_OK)
        else:
            return Response({"detail": "Invalid password reset link."}, status=status.HTTP_400_BAD_REQUEST)

class ManageRolesView(APIView):
    permission_classes = [permissions.IsAdminUser]
    serializer_class = ManageRolesSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data['email']
        role = serializer.validated_data['role']
        try:
            user = User.objects.get(email=email)
            user.role = role
            user.save()
            return Response({"detail": f"User role updated to {role}."}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({"detail": "User with this email does not exist."}, status=status.HTTP_404_NOT_FOUND)

class InstitutionView(generics.ListCreateAPIView):
    queryset = Institution.objects.all()
    serializer_class = InstitutionSerializer
    permission_classes = [permissions.IsAdminUser]

class UserView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

class PendingUsersView(generics.ListAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated, permissions.IsAdminUser]

    def get_queryset(self):
        if self.request.user.is_superuser:
            return User.objects.filter(is_active=False, is_email_verified=True)
        elif self.request.user.is_staff:
            return User.objects.filter(is_active=False, is_email_verified=True, institution=self.request.user.institution)
        return User.objects.none()


class ActivateUserByAdminView(APIView):
    permission_classes = [permissions.IsAuthenticated, permissions.IsAdminUser]

    def post(self, request, user_id):
        try:
            user = User.objects.get(id=user_id, is_active=False, is_email_verified=True)
            if request.user.is_superuser or (request.user.is_staff and request.user.institution == user.institution):
                user.is_active = True
                user.save()
                self.send_activation_notification(user)
                return Response({'message': 'User activated successfully'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'You do not have permission to activate this user'}, status=status.HTTP_403_FORBIDDEN)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    def send_activation_notification(self, user):
        context = {
            'user': user,
            'login_url': f"{settings.FRONTEND_URL}/login"
        }
        html_message = render_to_string('emails/account_activated_email_template.html', context)
        plain_message = strip_tags(html_message)
        send_mail(
            'Your GymWise account has been activated',
            plain_message,
            settings.DEFAULT_FROM_EMAIL,
            [user.email],
            html_message=html_message,
        )