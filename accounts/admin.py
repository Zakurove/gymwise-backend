from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _
from .models import User, Institution, Member, AIModel

class UserAdmin(BaseUserAdmin):
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        (_('Personal info'), {'fields': ('first_name', 'last_name')}),
        (_('Institutional info'), {'fields': ('institution', 'role')}),
        (_('Permissions'), {'fields': ('is_active', 'is_staff', 'is_superuser', 'is_email_verified', 'groups', 'user_permissions')}),
        (_('Important dates'), {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'first_name', 'last_name', 'institution', 'role', 'is_active', 'is_staff'),
        }),
    )
    list_display = ('email', 'first_name', 'last_name', 'institution', 'role', 'is_active', 'is_staff', 'is_email_verified')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'institution', 'role', 'is_email_verified')
    search_fields = ('email', 'first_name', 'last_name')
    ordering = ('email',)
    filter_horizontal = ('groups', 'user_permissions',)

class InstitutionAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_active', 'created_at')
    list_filter = ('is_active',)
    search_fields = ('name',)

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        return qs.filter(id=request.user.institution.id)

    def has_change_permission(self, request, obj=None):
        if not obj:
            return True
        return request.user.is_superuser or request.user.institution == obj

    def has_delete_permission(self, request, obj=None):
        if not obj:
            return True
        return request.user.is_superuser or request.user.institution == obj

class MemberAdmin(admin.ModelAdmin):
    list_display = ('user', 'institution')
    list_filter = ('institution',)
    search_fields = ('user__email', 'user__first_name', 'user__last_name')

class AIModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'institution', 'created_at', 'updated_at')
    list_filter = ('institution',)
    search_fields = ('name', 'institution__name')

admin.site.register(User, UserAdmin)
admin.site.register(Institution, InstitutionAdmin)
admin.site.register(Member, MemberAdmin)
admin.site.register(AIModel, AIModelAdmin)