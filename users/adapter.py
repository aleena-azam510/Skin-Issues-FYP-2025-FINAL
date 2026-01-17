# users/adapter.py

from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.shortcuts import redirect
from django.urls import reverse

class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    """
    Ensures that social signups are redirected to a complete profile page
    if the custom user model requires extra fields (like 'role').
    """

    def is_open_for_signup(self, request, sociallogin):
        # Prevent automatic login until we check for required fields
        # if you have required fields that are not covered by social login (like 'role')
        return True

    def save_user(self, request, sociallogin, form=None):
        """
        Saves the user and the social account.
        This is where the user object is typically created and saved first.
        """
        # Call the default implementation to create/save the user and account
        user = super().save_user(request, sociallogin, form)
        
        # --- Custom Logic for Role/Required Fields (Example) ---
        # If your CustomUser model has fields with null=False, 
        # you might need to set a default value here.
        if not user.role:
            user.role = 'user' # Set a mandatory default role
            user.save()
            
        return user