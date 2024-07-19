from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Booking
from .serializers import BookingSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.mail import send_mail
from django.conf import settings

@api_view(['POST'])
def book_demo(request):
    if request.method == 'POST':
        serializer = BookingSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
@api_view(['GET'])
def get_bookings(request):
    bookings = Booking.objects.all()
    serializer = BookingSerializer(bookings, many=True)
    return Response(serializer.data)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags

class ContactFormView(APIView):
    def post(self, request):
        name = request.data.get('name')
        email = request.data.get('email')
        message = request.data.get('message')
        
        if not all([name, email, message]):
            return Response({"error": "All fields are required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Prepare email content
            subject = f"New contact form submission from {name}"
            html_message = render_to_string('contact_email_template.html', {
                'name': name,
                'email': email,
                'message': message
            })
            plain_message = strip_tags(html_message)
            
            # Send email
            send_mail(
                subject,
                plain_message,
                'contact@gymwise.tech',
                ['contact@gymwise.tech'],
                html_message=html_message,
                fail_silently=False,
            )
            
            # Send confirmation email to the user
            confirmation_subject = "We've received your message - GymWise"
            confirmation_html = render_to_string('confirmation_email_template.html', {
                'name': name
            })
            confirmation_plain = strip_tags(confirmation_html)
            
            send_mail(
                confirmation_subject,
                confirmation_plain,
                'contact@gymwise.tech',
                [email],
                html_message=confirmation_html,
                fail_silently=False,
            )
            
            return Response({"message": "Your message has been sent successfully"}, status=status.HTTP_200_OK)
        
        except Exception as e:
            # Log the error (you should set up proper logging)
            print(f"Error sending email: {str(e)}")
            return Response({"error": "An error occurred while sending your message. Please try again later."}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)