from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    # Add custom fields to admin panel forms
    fieldsets = UserAdmin.fieldsets + (
        ('Custom Fields', {
            'fields': ('role', 'license_number', 'license_document', 'is_verified_doctor',
                       'admin_code', 'can_approve_doctors', 'access_level'),
        }),
    )
    
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Custom Fields', {
            'fields': ('role', 'license_number', 'license_document', 'is_verified_doctor',
                       'admin_code', 'can_approve_doctors', 'access_level'),
        }),
    )

admin.site.register(CustomUser, CustomUserAdmin)

from django.contrib import admin
from .models import MedicalResource

@admin.register(MedicalResource)
class MedicalResourceAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at')
    search_fields = ('title', 'description')
