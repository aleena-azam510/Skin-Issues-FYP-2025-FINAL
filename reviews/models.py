from django.db import models

# Create your models here.
# reviews/models.py (or predictor/models.py if you prefer)

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth import get_user_model

User = get_user_model() # Get the currently active user model

class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True,
                             help_text="The user who submitted the review (optional).")
    reviewer_name = models.CharField(max_length=100,
                                     help_text="Name of the person submitting the review.")
    rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="Rating given by the user (1 to 5 stars)."
    )
    review_text = models.TextField(
        help_text="The detailed text of the review."
    )
    created_at = models.DateTimeField(auto_now_add=True,
                                      help_text="Timestamp when the review was created.")

    class Meta:
        ordering = ['-created_at'] # Order reviews by most recent first

    def __str__(self):
        return f"Review by {self.reviewer_name} - {self.rating} stars"