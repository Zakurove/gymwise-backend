from django.contrib import admin
from .models import Booking

@admin.register(Booking)
class BookingAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'phone', 'preferred_date', 'created_at')
    search_fields = ('name', 'email', 'phone')
    list_filter = ('preferred_date', 'created_at')