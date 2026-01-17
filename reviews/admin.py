# reviews/admin.py

from django.contrib import admin
from .models import Review

@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('reviewer_name', 'rating', 'created_at', 'user', 'short_review_text')
    list_filter = ('rating', 'created_at')
    search_fields = ('reviewer_name', 'review_text')
    readonly_fields = ('user', 'reviewer_name', 'rating', 'review_text', 'created_at')

    def short_review_text(self, obj):
        return obj.review_text[:75] + '...' if len(obj.review_text) > 75 else obj.review_text
    short_review_text.short_description = 'Review Text'