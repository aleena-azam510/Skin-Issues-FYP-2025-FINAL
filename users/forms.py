from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser, SkinProgress, DoctorProfile, Appointment

# --------------------------
# USER SIGNUP FORM
# --------------------------
class CustomUserCreationForm(UserCreationForm):
    clerk_code = forms.CharField(max_length=50, required=False, help_text="Enter clerk verification code.")
    admin_code = forms.CharField(max_length=50, required=False)
    email = forms.EmailField(required=True)
    license_number = forms.CharField(max_length=100, required=False)
    license_document = forms.FileField(required=False)
    
    # ✅ New fields
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other')
    ]
    gender = forms.ChoiceField(choices=GENDER_CHOICES, required=True)
    age = forms.IntegerField(min_value=0, max_value=120, required=True)

    class Meta:
        model = CustomUser
        fields = (
            'username', 'email', 'password1', 'password2',
            'role', 'gender', 'age', 'license_number', 'license_document',
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            current_classes = field.widget.attrs.get('class', '')
            field.widget.attrs['class'] = f'{current_classes} input-compact-style'.strip()
            if isinstance(field.widget, (forms.TextInput, forms.EmailInput, forms.PasswordInput, forms.NumberInput)):
                field.widget.attrs.setdefault('placeholder', field.label)

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        role = cleaned_data.get("role")
        license_number = cleaned_data.get("license_number")
        license_document = cleaned_data.get("license_document")

        if role == 'doctor':
            if not license_number:
                self.add_error('license_number', "License number is required for doctors.")
            if not license_document:
                self.add_error('license_document', "License document file is required for doctors.")
        return cleaned_data


# --------------------------
# SKIN PROGRESS FORM
# --------------------------
class SkinProgressForm(forms.ModelForm):
    class Meta:
        model = SkinProgress
        fields = ['image']
        widgets = {
            'image': forms.ClearableFileInput(attrs={
                'id': 'id_image',
                'class': 'file-input',
                'accept': 'image/*'
            })
        }


# --------------------------
# APPOINTMENT FORM
# --------------------------
class AppointmentForm(forms.ModelForm):
    class Meta:
        model = Appointment
        fields = [
            "patient_name",
            "phone",
            "email",
            "date",
            "time",
            "reason",
        ]
        widgets = {
            "date": forms.DateInput(attrs={"type": "date"}),
            "time": forms.TimeInput(attrs={"type": "time"}),
            "reason": forms.Textarea(attrs={"rows": 3}),
        }


# --------------------------
# DOCTOR PROFILE FORM
# --------------------------
DAYS_OF_WEEK = [
    ("Monday", "Monday"),
    ("Tuesday", "Tuesday"),
    ("Wednesday", "Wednesday"),
    ("Thursday", "Thursday"),
    ("Friday", "Friday"),
    ("Saturday", "Saturday"),
    ("Sunday", "Sunday"),
]

class DoctorProfileForm(forms.ModelForm):
    # Dynamically add availability fields
    for day in DAYS_OF_WEEK:
        locals()[f"{day[0].lower()}_times"] = forms.CharField(
            required=False,
            label=f"{day[0]} Availability",
            widget=forms.TextInput(attrs={
                "placeholder": "09:00, 10:00, 14:30 etc."
            })
        )

    class Meta:
        model = DoctorProfile
        fields = [
            "specialization", "experience_years", "bio",
            "profile_image", "clinic_address", "phone_number",
            "consultation_fee",
            "bank_name", "account_number", "account_holder_name", "upi_id",
        ]
        widgets = {
            "bio": forms.Textarea(attrs={"rows": 4}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.availability:
            for day in DAYS_OF_WEEK:
                field_name = f"{day[0].lower()}_times"
                self.fields[field_name].initial = ", ".join(
                    self.instance.availability.get(day[0], [])
                )

    def clean(self):
        cleaned_data = super().clean()
        availability = {}
        for day in DAYS_OF_WEEK:
            times_str = cleaned_data.get(f"{day[0].lower()}_times", "")
            times_list = [t.strip() for t in times_str.split(",") if t.strip()]
            availability[day[0]] = times_list
        cleaned_data['availability'] = availability
        return cleaned_data

    def save(self, commit=True):
        obj = super().save(commit=False)
        obj.availability = self.cleaned_data['availability']
        if commit:
            obj.save()
        return obj
