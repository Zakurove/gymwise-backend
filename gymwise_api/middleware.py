# gymwise-backend/gymwise_api/middleware.py

import logging
from django.http import JsonResponse

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_exception(self, request, exception):
        logger.exception(f"Unhandled exception: {str(exception)}")
        return JsonResponse({
            'error': 'An unexpected error occurred',
            'details': str(exception)
        }, status=500)