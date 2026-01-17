# from django.db import models

# # Create your models here.

from django.db import models
from django.utils.html import format_html
from django.utils.text import slugify # Don't forget to import slugify if you're using slugs


class SkinCondition(models.Model):
    name = models.CharField(max_length=100, unique=True)
    causes = models.TextField()
    symptoms = models.TextField()

    def __str__(self):
        return self.name 

class Remedy(models.Model):
    APPROACH_CHOICES = [
        ("ayurvedic", "Ayurvedic"),
        ("western", "Western Home Care"),
        ("herbal", "Herbal Only"),
    ]

    SAFETY_LEVEL_CHOICES = [
        ("safe", "Safe"),
        ("caution", "Use with caution"),
        ("avoid_pregnancy", "Avoid during pregnancy"),
    ]
    CATEGORY_CHOICES = [
        ("home", "Home Remedy"),
        ("medical", "Medical Treatment"),
        ("lifestyle", "Lifestyle Change"),
    ]
    skin_condition = models.ForeignKey(SkinCondition, on_delete=models.CASCADE, related_name='remedy_set')

    title = models.CharField(max_length=100)
    amount = models.CharField(max_length=100)
    directions = models.TextField()

    approach = models.CharField(max_length=20, choices=APPROACH_CHOICES, default="herbal")
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default="home")
    traditional_evidence = models.PositiveIntegerField(default=3)  # 1-5 stars
    scientific_evidence = models.PositiveIntegerField(default=2)  # 1-5 stars
    safety_level = models.CharField(max_length=20, choices=SAFETY_LEVEL_CHOICES, default="safe")

    image = models.ImageField(upload_to='remedy_images/', blank=True, null=True)

    def get_image_url(self):
        return self.image.url if self.image else None

    def formatted_directions(self):
        if not self.directions:
            return []
        all_steps = []
        for line in self.directions.split('\n'):
            all_steps.extend([step.strip() for step in line.split(',') if step.strip()])
        return all_steps

    def image_preview(self):
        if self.image:
            return format_html('<img src="{}" style="max-height: 100px; max-width: 100px;" />', self.image.url)
        return "No image"

    image_preview.short_description = 'Preview'

    def __str__(self):
        return f"{self.skin_condition.name} → {self.title}"


class FollowUpQuestion(models.Model):
    skin_condition = models.ForeignKey(SkinCondition, on_delete=models.CASCADE, related_name='followup_questions')
    question = models.TextField()
    symptom_key = models.CharField(max_length=100)
    why_this_question = models.TextField(
        help_text="Explain why this question matters (e.g., to differentiate similar conditions)."
    )

    def __str__(self):
        return f"{self.skin_condition.name} → {self.question[:50]}"
    
from django.utils import timezone
class Treatment(models.Model):
    skin_condition = models.ForeignKey(
        SkinCondition,
        on_delete=models.CASCADE,
        null=True,   # <-- temporarily allow nulls
        blank=True
    )
    title = models.CharField(max_length=255)
    category = models.CharField(max_length=50, choices=[('medical','Medical'), ('home','Home')])
    scientific_evidence = models.IntegerField(default=0)  # renamed from confidence
    amount = models.CharField(max_length=255, blank=True, null=True)
    directions = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    image = models.ImageField(upload_to='treatments/', null=True, blank=True)

    def __str__(self):
        return f"{self.title} ({self.skin_condition})"

class PersonalizedPlan(models.Model):
    condition_name = models.CharField(max_length=100, unique=True)
    
    # Add detailed sections
    diet = models.TextField()
    skincare = models.TextField()
    exercise = models.TextField()
    sleep = models.TextField()
    
    created_at = models.DateTimeField(auto_now_add=True)  # <-- new field
    def __str__(self):
        return self.condition_name
# models.py (conceptual)
from django.db import models

from django.db import models
from django.utils.text import slugify
from django.core.serializers.json import DjangoJSONEncoder

from django.db import models
from django.utils.text import slugify
from django.core.serializers.json import DjangoJSONEncoder # Make sure this import is present

from django.db import models
from django.utils.text import slugify
from django.core.serializers.json import DjangoJSONEncoder # Make sure this is imported

class SkinCondition_page(models.Model):
    # Basic SEO Fields
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True, blank=True, max_length=255)
    meta_description = models.TextField()
    meta_keywords = models.CharField(max_length=255)

    # Card Images
    main_card_image = models.ImageField(upload_to='skin_conditions/card_images/', blank=True, null=True)
    basics_card_image = models.ImageField(upload_to='skin_conditions/card_images/basics/', blank=True, null=True)
    symptoms_card_image = models.ImageField(upload_to='skin_conditions/card_images/symptoms/', blank=True, null=True) # New card image
    causes_card_image = models.ImageField(upload_to='skin_conditions/card_images/causes/', blank=True, null=True)
    treatments_card_image = models.ImageField(upload_to='skin_conditions/card_images/treatments/', blank=True, null=True)
    doctor_card_image = models.ImageField(upload_to='skin_conditions/card_images/doctor/', blank=True, null=True)
    

    # Basics Section
    basics_title = models.CharField(max_length=200)
    basics_summary = models.TextField()
    basics_content = models.TextField(help_text="HTML content with <ul>, <li>, <strong> etc.")
    basics_image = models.ImageField(upload_to='skin_conditions/basics_images/', blank=True, null=True)

    # Symptoms Section (NEW SECTION)
    symptoms_title = models.CharField(max_length=200, blank=True, null=True)
    symptoms_summary = models.TextField(blank=True, null=True)
    symptoms_content = models.TextField(
        blank=True, 
        null=True, 
        help_text="HTML content describing symptoms, e.g., using <ul>, <li>, <strong>, etc. Can also be a general paragraph."
    )
    symptoms_image = models.ImageField(upload_to='skin_conditions/symptoms_images/', blank=True, null=True)

    # Causes Section (Enhanced with icon support)
    causes_title = models.CharField(max_length=200)
    causes_summary = models.TextField()
    causes_intro = models.TextField(blank=True)
    causes_details = models.JSONField(
        blank=True, 
        null=True, 
        encoder=DjangoJSONEncoder,
        help_text="JSON format: [{'heading': '...', 'description': '...', 'icon': 'fas fa-icon-name'}, ...]"
    )
    causes_image = models.ImageField(upload_to='skin_conditions/causes_images/', blank=True, null=True)

    # Treatments Section (Enhanced with icon support)
    treatments_title = models.CharField(max_length=200)
    treatments_summary = models.TextField()
    treatments_details = models.JSONField(
        blank=True,
        null=True,
        encoder=DjangoJSONEncoder,
        help_text="JSON format: [{'heading': '...', 'description': '...', 'icon': 'fas fa-icon-name'}, ...]"
    )
    treatments_image = models.ImageField(upload_to='skin_conditions/treatments_images/', blank=True, null=True)

    # Prevention Section
    prevention_title = models.CharField(max_length=200, blank=True, null=True)
    prevention_summary = models.TextField(blank=True, null=True) # <--- ADD THIS LINE
    prevention_skincare = models.JSONField(
        blank=True,
        null=True,
        encoder=DjangoJSONEncoder,
        help_text="JSON format: [{'text': '...', 'icon': 'fas fa-icon-name'}, ...]"
    )
    prevention_lifestyle = models.JSONField(
        blank=True,
        null=True,
        encoder=DjangoJSONEncoder,
        help_text="JSON format: [{'text': '...', 'icon': 'fas fa-icon-name'}, ...]"
    )
    prevention_card_image = models.ImageField(upload_to='skin_conditions/card_images/prevention/', blank=True, null=True)

    prevention_image = models.ImageField(upload_to='skin_conditions/prevention_images/', blank=True, null=True)

    # When to See a Doctor Section
    doctor_title = models.CharField(max_length=200, blank=True, null=True)
    doctor_summary = models.TextField(blank=True, null=True)
    doctor_details = models.JSONField(
        blank=True,
        null=True,
        encoder=DjangoJSONEncoder,
        help_text="JSON format: [{'point': '...', 'icon': 'fas fa-icon-name'}, ...]"
    )
    doctor_image = models.ImageField(upload_to='skin_conditions/doctor_images/', blank=True, null=True)

    # Automatic slug generation
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title

    # Helper methods for default icons
    @staticmethod
    def get_default_icon(heading):
        """Returns appropriate Font Awesome icon based on heading keywords"""
        heading_lower = heading.lower()
        icon_mapping = {
            'hormon': 'fa-hand-holding-medical',
            'environ': 'fa-cloud-sun',
            'diet': 'fa-utensils',
            'stress': 'fa-brain',
            'bacter': 'fa-bacteria',
            'genet': 'fa-dna',
            'oil': 'fa-oil-can',
            'skin': 'fa-spa',
            'clean': 'fa-soap',
            'sleep': 'fa-moon',
            'water': 'fa-tint',
            'exercise': 'fa-running',
            'pain': 'fa-hand-holding-medical',
            'spread': 'fa-expand-arrows-alt',
            'severe': 'fa-exclamation-triangle',
            'persist': 'fa-history',
            'worsen': 'fa-long-arrow-alt-up',
            'infection': 'fa-bacteria',
            'blister': 'fa-burn',
            'fever': 'fa-thermometer-half',
            'redness': 'fa-tint',
            'swelling': 'fa-hand-holding-medical',
            'pus': 'fa-syringe',
            'unknown': 'fa-question-circle',
            'itch': 'fa-itch', # Common symptom
            'rash': 'fa-allergies', # Common symptom
            'dry': 'fa-leaf', # Can represent dry skin
            'flake': 'fa-eraser', # Flaking skin
            'patch': 'fa-border-none', # Patches on skin
            'bump': 'fa-dot-circle', # Bumps on skin
            'discolor': 'fa-palette', # Skin discoloration
            'burning': 'fa-fire', # Burning sensation
        }
        
        for key, icon in icon_mapping.items():
            if key in heading_lower:
                return f'fas {icon}'
        return 'fas fa-info-circle'

    @property
    def causes_with_icons(self):
        """Ensures all causes have appropriate icons"""
        if not self.causes_details:
            return []
            
        return [
            {
                'heading': item['heading'],
                'description': item['description'],
                'icon': item.get('icon') or self.get_default_icon(item['heading'])
            }
            for item in self.causes_details
        ]

    @property
    def treatments_with_icons(self):
        """Ensures all treatments have appropriate icons"""
        if not self.treatments_details:
            return []
            
        return [
            {
                'heading': item['heading'],
                'description': item['description'],
                'icon': item.get('icon') or self.get_default_icon(item['heading'])
            }
            for item in self.treatments_details
        ]

    @property
    def doctor_details_with_icons(self):
        """Ensures all 'When to See a Doctor' details have appropriate icons"""
        if not self.doctor_details:
            return []
            
        return [
            {
                'point': item['point'],
                'icon': item.get('icon') or self.get_default_icon(item['point'])
            }
            for item in self.doctor_details
        ]
        
# models.py
# models.py
from django.db import models
from django.template.defaultfilters import slugify
from django.contrib.auth import get_user_model # Recommended way to get the User model
import json # You'll need this if you want to set a default for JSONField, though default=dict is often better

# Get the custom User model if you have one, otherwise it defaults to django.contrib.auth.models.User
User = get_user_model()

class Article(models.Model):
    CATEGORY_CHOICES = [
        ('remedies', 'DIY Beauty Lab'),
        ('prevention', 'Holistic Skin Wisdom'),
        ('ingredients', "Nature's Pharmacy"),
        ('secret', "Skin Secrets"),
    ]

    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True, max_length=255, blank=True)
    author = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    featured_image = models.ImageField(upload_to='articles/', null=True, blank=True)
    excerpt = models.TextField(max_length=600)
    
    # Final state for the content JSONField
    # default=dict ensures new articles have an empty JSON object
    # null=True and blank=True are no longer strictly needed for a JSONField with a default,
    # but I'm including them as they align with your SkinCondition_page model and allow flexibility.
    content = models.JSONField(default=dict, blank=True, null=True)# TEMPORARY: for next makemigrations

    published_date = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    is_featured = models.BooleanField(default=False)
    reading_time = models.IntegerField(default=5)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-published_date']
        verbose_name = "Article"
        verbose_name_plural = "Articles"
        
