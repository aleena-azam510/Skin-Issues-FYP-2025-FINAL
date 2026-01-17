# chatbot/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('chatbot/', views.chatbot, name='chatbot'),
    # path('chatbot/greeting/', views.initial_greeting, name='initial_greeting'), # Add this line
    # path("chatbot/suggestions/", views.search_suggestions, name="chatbot-suggestions"),
]

