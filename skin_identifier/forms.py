from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator

class SkinTypeForm(forms.Form):
    # Basic User Information
    name = forms.CharField(
        label="Full Name",
        max_length=100,
        widget=forms.TextInput(attrs={'placeholder': 'Your full name'})
    )
    email = forms.EmailField(
        label="Email Address",
        widget=forms.EmailInput(attrs={'placeholder': 'Your email address'})
    )
    age = forms.IntegerField(
        label="Age",
        validators=[MinValueValidator(13), MaxValueValidator(99)],
        widget=forms.NumberInput(attrs={'placeholder': 'Your age'})
    )
    GENDER_CHOICES = [
        ('', 'Select your gender'),  # Empty value for placeholder
        ('female', 'Female'),
        ('male', 'Male'),
        ('other', 'Other'),
        ('prefer_not_to_say', 'Prefer not to say'),
    ]
    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        label="Gender",
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    # --- Expanded Skin Characteristics ---
    SKIN_FEEL_CHOICES = [
        ('tight', 'Tight and dry'),
        ('comfortable', 'Comfortable and balanced'),
        ('shiny', 'Shiny and oily'),
        ('combination', 'Dry in some areas, oily in others')
    ]
    skin_feel = forms.ChoiceField(
        choices=SKIN_FEEL_CHOICES,
        widget=forms.RadioSelect,
        label="How does your skin typically feel a few hours after cleansing?"
    )

    SKIN_CONCERNS_CHOICES = [
        ('acne', 'Acne or breakouts'),
        ('blackheads', 'Blackheads or enlarged pores'),
        ('dryness', 'Dryness or flakiness'),
        ('redness', 'Redness or irritation'),
        ('aging', 'Fine lines or wrinkles'),
        ('dark_spots', 'Dark spots or hyperpigmentation'),
        ('dullness', 'Dullness or uneven texture'),
        ('sensitivity', 'Sensitivity')
    ]
    skin_concerns = forms.MultipleChoiceField(
        choices=SKIN_CONCERNS_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        label="Which of these skin concerns do you experience?",
        required=False
    )

    SENSITIVITY_CHOICES = [
        ('not_sensitive', 'Not sensitive'),
        ('somewhat_sensitive', 'Somewhat sensitive'),
        ('very_sensitive', 'Very sensitive')
    ]
    sensitivity_level = forms.ChoiceField(
        choices=SENSITIVITY_CHOICES,
        widget=forms.RadioSelect,
        label="How sensitive is your skin?"
    )

    PORE_SIZE_CHOICES = [
        ('small', 'Small, barely visible'),
        ('medium', 'Medium, somewhat visible'),
        ('large', 'Large, clearly visible')
    ]
    pore_size = forms.ChoiceField(
        choices=PORE_SIZE_CHOICES,
        widget=forms.RadioSelect,
        label="Which of these best matches your skin's pore size?"
    )
    
    # --- New and Expanded Questions for More Diverse Data ---

    # New field: Skin texture and appearance
    SKIN_TEXTURE_CHOICES = [
        ('smooth', 'Smooth and even'),
        ('uneven_rough', 'Uneven or rough to the touch'),
        ('bumpy_clogged', 'Bumpy with clogged pores'),
        ('flaky_patchy', 'Flaky or patchy')
    ]
    skin_texture = forms.ChoiceField(
        choices=SKIN_TEXTURE_CHOICES,
        widget=forms.RadioSelect,
        label="Describe the texture of your skin."
    )

    # New field: Environmental factors
    CLIMATE_CHOICES = [
        ('dry', 'Dry or arid'),
        ('humid', 'Humid'),
        ('cold', 'Cold and windy'),
        ('temperate', 'Temperate (balanced)'),
        ('mixed', 'Mixed seasons')
    ]
    climate = forms.ChoiceField(
        choices=CLIMATE_CHOICES,
        widget=forms.RadioSelect,
        label="What is the general climate where you live?"
    )

    # New field: Dietary habits
    DIET_CHOICES = [
        ('balanced', 'Balanced, includes fruits and vegetables'),
        ('sugary', 'High in sugar and processed foods'),
        ('dairy', 'High in dairy products'),
        ('spicy_oily', 'High in spicy or oily foods')
    ]
    diet_factors = forms.MultipleChoiceField(
        choices=DIET_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        label="Which of these dietary habits describe you?",
        required=False
    )

    # New field: Hormonal influences (important for adult acne)
    HORMONAL_CHOICES = [
        ('consistent', 'Skin is consistent throughout the month'),
        ('pre_cycle', 'Breakouts/oiliness get worse before my menstrual cycle'),
        ('stress_related', 'Breakouts seem stress-related'),
        ('menopause', 'Experiencing changes due to menopause')
    ]
    hormonal_factors = forms.MultipleChoiceField(
        choices=HORMONAL_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        label="Do you notice any hormonal patterns in your skin?",
        required=False
    )

    # New field: Skincare routine longevity
    ROUTINE_LENGTH_CHOICES = [
        ('less_3_months', 'Less than 3 months'),
        ('3_to_12_months', '3-12 months'),
        ('1_to_3_years', '1-3 years'),
        ('more_3_years', 'More than 3 years')
    ]
    skincare_routine_length = forms.ChoiceField(
        choices=ROUTINE_LENGTH_CHOICES,
        widget=forms.RadioSelect,
        label="How long have you been following your current skincare routine?",
        required=False  # <--- Add this line to make the field optional
)

    current_routine = forms.CharField(
        label="Describe your current skincare routine (products you use and frequency)",
        widget=forms.Textarea(attrs={
            'placeholder': 'e.g. I cleanse with a foaming cleanser twice daily, use a vitamin C serum in the morning...'
        }),
        required=False
    )

    LIFESTYLE_CHOICES = [
        ('stress', 'High stress levels'),
        ('smoking', 'Smoke or are exposed to smoke'),
        ('alcohol', 'Regular alcohol consumption'),
        ('exercise', 'Regular exercise (3+ times/week)'),
        ('sleep', 'Consistently get 7-8 hours sleep'),
        ('sun', 'Frequent sun exposure without protection'),
        ('pollution', 'Live in a high pollution area')
    ]
    lifestyle_factors = forms.MultipleChoiceField(
        choices=LIFESTYLE_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        label="Which of these lifestyle factors apply to you?",
        required=False
    )

    PRODUCT_REACTION_CHOICES = [
        ('none', 'No reaction, tolerates most products'),
        ('irritation', 'Sometimes experiences irritation'),
        ('redness', 'Often gets redness or stinging'),
        ('breakouts', 'Frequently breaks out from new products'),
        ('dryness', 'Becomes dry or flaky')
    ]
    product_reaction = forms.ChoiceField(
        choices=PRODUCT_REACTION_CHOICES,
        widget=forms.RadioSelect,
        label="How does your skin typically react to new skincare products?"
    )

    def clean_age(self):
        age = self.cleaned_data['age']
        if age < 13:
            raise forms.ValidationError("You must be at least 13 years old to use this service.")
        return age