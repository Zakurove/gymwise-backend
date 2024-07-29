from django.db import connection
from .models import Institution

class TenantMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        hostname = request.get_host().split(':')[0].lower()
        subdomain = hostname.split('.')[0]
        try:
            request.tenant = Institution.objects.get(subdomain=subdomain)
        except Institution.DoesNotExist:
            request.tenant = None

        response = self.get_response(request)
        return response