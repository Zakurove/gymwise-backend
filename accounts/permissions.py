from rest_framework import permissions

class IsAdminUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user and (request.user.role == 'admin' or request.user.role == 'superadmin')

class IsSuperAdminOrAdmin(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user and request.user.role in ['superadmin', 'admin']