from django import template
from django.utils.html import strip_tags
from django.template.defaultfilters import truncatechars

register = template.Library()

@register.filter
def num_range(value):
    """
    Filter to create a list of numbers from 0 to value-1.
    Usage: {% for i in 5|num_range %}
    """
    return range(value)

@register.filter
def create_star_list(value):
    """
    Generates a list of booleans for solid (True) and empty (False) stars.
    e.g., 4 would return [True, True, True, True, False] (out of 5)
    """
    stars = []
    # Ensure value is an integer and within the valid range (1-5)
    rating = max(1, min(5, int(value))) if value is not None else 0

    for i in range(1, 6): # Assuming rating is out of 5
        stars.append(i <= rating)
    return stars

@register.filter
def truncate_review_text(value, arg=100): # Default to 100 characters
    """
    Truncates a string after a certain number of characters and adds an ellipsis.
    Also strips HTML tags.
    Usage: {{ review.review_text|truncate_review_text:50 }}
    """
    return truncatechars(strip_tags(value), arg)