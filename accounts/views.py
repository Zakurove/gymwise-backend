from rest_framework import status, generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from .serializers import (
    RegisterSerializer, ActivateUserSerializer, ForgotPasswordSerializer, 
    ResetPasswordSerializer, ManageRolesSerializer, InstitutionSerializer,
    UserSerializer
)
from .models import User, Institution
from django.core.mail import send_mail
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_str
from rest_framework_simplejwt.views import TokenObtainPairView
import traceback
from django.contrib.auth import authenticate
from .permissions import IsAdminUser, IsSuperAdminOrAdmin
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes, force_str
from django.template.loader import render_to_string
from django.utils.html import strip_tags

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]

class ActivateUserView(APIView):
    def get(self, request, uidb64, token):
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            user.is_email_verified = True
            user.save()
            return Response({'message': 'Email verified successfully. Awaiting admin approval.'}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Activation link is invalid!'}, status=status.HTTP_400_BAD_REQUEST)

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

class ManageRolesView(APIView):
    permission_classes = [IsSuperAdminOrAdmin]

    def post(self, request):
        serializer = ManageRolesSerializer(data=request.data)
        if serializer.is_valid():
            try:
                user = User.objects.get(email=serializer.validated_data['email'])
                if request.user.role == 'superadmin' or (request.user.role == 'admin' and request.user.institution == user.institution):
                    user.role = serializer.validated_data['role']
                    user.save()
                    return Response({'message': 'User role updated'}, status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'You do not have permission to manage this user'}, status=status.HTTP_403_FORBIDDEN)
            except User.DoesNotExist:
                return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class InstitutionView(generics.ListCreateAPIView):
    queryset = Institution.objects.all()
    serializer_class = InstitutionSerializer
    permission_classes = [IsAdminUser]

class CustomTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        try:
            email = request.data.get('email')
            password = request.data.get('password')
            
            user = authenticate(request, email=email, password=password)
            
            if user is None:
                return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
            
            if not user.is_active:
                return Response({"error": "User account is not active"}, status=status.HTTP_401_UNAUTHORIZED)
            
            response = super().post(request, *args, **kwargs)
            
            # Add user info to response
            user_serializer = UserSerializer(user)
            response.data['user'] = user_serializer.data
            
            return response
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

class PendingUsersView(generics.ListAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated, IsAdminUser]

    def get_queryset(self):
        if self.request.user.role == 'superadmin':
            return User.objects.filter(is_active=False, is_email_verified=True)
        elif self.request.user.role == 'admin':
            return User.objects.filter(is_active=False, is_email_verified=True, institution=self.request.user.institution)
        return User.objects.none()

class ActivateUserByAdminView(APIView):
    permission_classes = [IsAuthenticated, IsAdminUser]

    def post(self, request, user_id):
        try:
            user = User.objects.get(id=user_id, is_active=False, is_email_verified=True)
            if request.user.role == 'superadmin' or (request.user.role == 'admin' and request.user.institution == user.institution):
                user.is_active = True
                user.save()
                
                # Send activation notification email
                self.send_activation_notification(user)
                
                return Response({'message': 'User activated successfully'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'You do not have permission to activate this user'}, status=status.HTTP_403_FORBIDDEN)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    def send_activation_notification(self, user):
        context = {
            'user': user,
            'login_url': 'http://localhost:3000/login'  # Update this with your frontend login URL
        }
        html_message = render_to_string('emails/account_activated_email_template.html', context)
        plain_message = strip_tags(html_message)

        send_mail(
            'Your GymWise account has been activated',
            plain_message,
            'contact@gymwise.tech',
            [user.email],
            html_message=html_message,
            fail_silently=False,
        )