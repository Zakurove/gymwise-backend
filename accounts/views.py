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
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes, force_str
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
from .permissions import IsAdminUser, IsSuperAdminOrAdmin, IsSuperAdmin
from rest_framework.permissions import IsAuthenticated
import logging

logger = logging.getLogger(__name__)

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

    def post(self, request):
        serializer = ForgotPasswordSerializer(data=request.data)
        if serializer.is_valid():
            try:
                user = User.objects.get(email=serializer.validated_data['email'])
                token = default_token_generator.make_token(user)
                uid = urlsafe_base64_encode(force_bytes(user.pk))
                reset_link = f"http://localhost:3000/reset-password/{uid}/{token}/"

                context = {
                    'reset_link': reset_link,
                }
                html_message = render_to_string('emails/password_reset_email_template.html', context)
                plain_message = strip_tags(html_message)

                send_mail(
                    'Reset your GymWise password',
                    plain_message,
                    'contact@gymwise.tech',
                    [user.email],
                    html_message=html_message,
                    fail_silently=False,
                )
                return Response({'message': 'Password reset link sent'}, status=status.HTTP_200_OK)
            except User.DoesNotExist:
                return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ResetPasswordView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, uid, token):
        try:
            uid = force_str(urlsafe_base64_decode(uid))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            serializer = ResetPasswordSerializer(data=request.data)
            if serializer.is_valid():
                user.set_password(serializer.validated_data['password'])
                user.save()
                return Response({'message': 'Password reset successfully'}, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({'error': 'Reset link is invalid!'}, status=status.HTTP_400_BAD_REQUEST)
        
class InstitutionUsersView(generics.ListAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]

    def get_queryset(self):
        logger.info(f"User {self.request.user.id} fetching institution users")
        if self.request.user.role == 'superadmin':
            return User.objects.all()
        elif self.request.user.role == 'admin':
            return User.objects.filter(institution=self.request.user.institution)
        return User.objects.none()

class ManageRolesView(generics.GenericAPIView):
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]
    queryset = User.objects.all()
    serializer_class = UserSerializer

    def post(self, request, *args, **kwargs):
        user_id = request.data.get('user_id')
        new_role = request.data.get('role')

        if not user_id or not new_role:
            return Response({'error': 'User ID and new role are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user_to_update = User.objects.get(id=user_id)
            
            # Check permissions
            if request.user.role != 'superadmin' and (user_to_update.role == 'superadmin' or new_role == 'superadmin'):
                return Response({'error': 'You do not have permission to change superadmin roles'}, status=status.HTTP_403_FORBIDDEN)

            if request.user.role == 'admin' and user_to_update.institution != request.user.institution:
                return Response({'error': 'You can only change roles for users in your institution'}, status=status.HTTP_403_FORBIDDEN)

            user_to_update.role = new_role
            user_to_update.save()

            logger.info(f"User {user_to_update.id} role updated to {new_role} by admin {request.user.id}")
            return Response({'message': f"User role updated to {new_role}"}, status=status.HTTP_200_OK)

        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error in ManageRolesView: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class InstitutionView(generics.ListCreateAPIView):
    queryset = Institution.objects.all()
    serializer_class = InstitutionSerializer
    permission_classes = [IsSuperAdmin]

class UserView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

class PendingUsersView(generics.ListAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]

    def get_queryset(self):
        if self.request.user.role == 'superadmin':
            return User.objects.filter(is_active=False, is_email_verified=True)
        elif self.request.user.role == 'admin':
            return User.objects.filter(is_active=False, is_email_verified=True, institution=self.request.user.institution)
        return User.objects.none()

    def get_queryset(self):
        logger.info(f"User {self.request.user.id} attempting to fetch pending users")
        if self.request.user.role == 'superadmin':
            logger.info("Superadmin fetching all pending users")
            return User.objects.filter(is_active=False, is_email_verified=True)
        elif self.request.user.role == 'admin':
            logger.info(f"Admin user fetching pending users for institution {self.request.user.institution.id}")
            return User.objects.filter(is_active=False, is_email_verified=True, institution=self.request.user.institution)
        logger.warning(f"User {self.request.user.id} does not have permission to view pending users")
        return User.objects.none()

    def list(self, request, *args, **kwargs):
        try:
            queryset = self.get_queryset()
            serializer = self.get_serializer(queryset, many=True)
            logger.info(f"Successfully fetched {len(serializer.data)} pending users")
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Error in PendingUsersView: {str(e)}")
            return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ActivateUserByAdminView(generics.UpdateAPIView):
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]
    queryset = User.objects.all()
    lookup_url_kwarg = 'user_id'

    def update(self, request, *args, **kwargs):
        try:
            user = self.get_object()
            if request.user.role == 'superadmin' or (request.user.role == 'admin' and request.user.institution == user.institution):
                user.is_active = True
                user.save()
                logger.info(f"User {user.id} activated successfully by admin {request.user.id}")
                return Response({'message': 'User activated successfully'}, status=status.HTTP_200_OK)
            else:
                logger.warning(f"User {request.user.id} attempted to activate user {user.id} without permission")
                return Response({'error': 'You do not have permission to activate this user'}, status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            logger.error(f"Error in ActivateUserByAdminView: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

# Add this new view to get the current user's information
class CurrentUserView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = UserSerializer

    def get_object(self):
        return self.request.user


