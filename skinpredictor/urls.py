"""
URL configuration for skinpredictor project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from reviews import views as review_views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictor.urls')),
    # 1. Include allauth URLs for social authentication
    path('accounts/', include('allauth.urls')),  # <-- IMPORTANT: Adds Google, FB, Github flow
    path('', include('users.urls')),
    path('skin/', include('skin_identifier.urls')),
     # Include the reviews app URLs
    path('reviews/', include('reviews.urls')), # Or 'predictor.urls' if you put it there
    path('contact/submit/', review_views.contact_form_submit, name='contact_submit'),
    path('', include('chatbot.urls')),
    path('oauth/', include('social_django.urls', namespace='social')),
    # path('api/chatbot_api_gemini/', include('chatbot_api.urls')),


]
