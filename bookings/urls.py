from django.urls import path, include
from .views import book_demo, get_bookings, ContactFormView

urlpatterns = [
    path('book-demo/', book_demo, name='book-demo'),
    path('bookings/', get_bookings, name='get-bookings'),
    path('contact/', ContactFormView.as_view(), name='contact_form'),
]