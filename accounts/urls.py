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
    ActivateUserByAdminView,
    CurrentUserView,
    InstitutionUsersView,
)

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('activate/<str:uidb64>/<str:token>/', ActivateUserView.as_view(), name='activate'),
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('forgot-password/', ForgotPasswordView.as_view(), name='forgot_password'),
    path('reset-password/<str:uid>/<str:token>/', ResetPasswordView.as_view(), name='reset_password'),
    path('manage-roles/', ManageRolesView.as_view(), name='manage_roles'),
    path('user/', CurrentUserView.as_view(), name='current_user'),
    path('institution-users/', InstitutionUsersView.as_view(), name='institution_users'),
    path('activate-user/<int:user_id>/', ActivateUserByAdminView.as_view(), name='activate_user'),
    # path('tenant/<str:subdomain>/', get_tenant_by_subdomain, name='get_tenant_by_subdomain'),
]

