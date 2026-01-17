# reviews/urls.py (or predictor/urls.py)

from django.urls import path
from . import views

app_name = 'reviews' # Namespace for your app's URLs

urlpatterns = [
    path('submit-review/', views.submit_review, name='submit_review'),
    path('get-reviews/', views.get_reviews_list, name='get_reviews_list'), # New URL
]