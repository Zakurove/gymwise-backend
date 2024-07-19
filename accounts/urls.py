from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views import (
    RegisterView,
    ActivateUserView,
    ForgotPasswordView,
    ResetPasswordView,
    ManageRolesView,
    InstitutionView,
    CustomTokenObtainPairView,
    UserView,
    PendingUsersView,
    ActivateUserByAdminView,
)

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('activate/<str:uidb64>/<str:token>/', ActivateUserView.as_view(), name='activate'),
    path('forgot-password/', ForgotPasswordView.as_view(), name='forgot-password'),
    path('reset-password/<str:uid>/<str:token>/', ResetPasswordView.as_view(), name='reset-password'),
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('manage-roles/', ManageRolesView.as_view(), name='manage_roles'),
    path('institutions/', InstitutionView.as_view(), name='institutions'),
    path('user/', UserView.as_view(), name='user'),
    path('pending-users/', PendingUsersView.as_view(), name='pending_users'),
    path('activate-user/<int:user_id>/', ActivateUserByAdminView.as_view(), name='activate_user'),
]