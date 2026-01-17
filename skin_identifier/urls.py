from django.urls import path
from .views import skin_identifier_view
from . import views  # Assuming views.py is in the same directory
from . import views  # Assuming views.py is in the same directory
from . import views  # Assuming views.py is in the same directory


urlpatterns = [
     path('skin-type/', skin_identifier_view, name='skin_identifier'),
     path('skin_identifier/', views.skin_identifier_view, name='skin_identifier_view'),

]
