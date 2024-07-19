from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, Institution

class UserAdmin(BaseUserAdmin):
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name')}),
        ('Institutional info', {'fields': ('institution', 'role')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'is_email_verified')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'first_name', 'last_name', 'institution', 'role'),
        }),
    )
    list_display = ('email', 'first_name', 'last_name', 'institution', 'role', 'is_active', 'is_staff', 'is_email_verified')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'institution', 'role', 'is_email_verified')
    search_fields = ('email', 'first_name', 'last_name')
    ordering = ('email',)
    filter_horizontal = ()

class InstitutionAdmin(admin.ModelAdmin):
    list_display = ('name',)
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

admin.site.register(User, UserAdmin)
admin.site.register(Institution, InstitutionAdmin)