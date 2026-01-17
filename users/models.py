from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings
from django.utils import timezone




class CustomUser(AbstractUser):
    ROLE_CHOICES = [
        ('user', 'Regular User'),
        ('doctor', 'Dermatologist'),
        ('admin', 'Administrator'),
    ]

    VERIFICATION_STATUS = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default='user'
    )
    # Gender & Age for users
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
    ]

    gender = models.CharField(max_length=16, choices=GENDER_CHOICES, blank=True, null=True)
    age = models.PositiveIntegerField(blank=True, null=True)

    # ------------------
    # Doctor fields
    # ------------------
    license_number = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    license_document = models.FileField(
        upload_to='license_docs/',
        blank=True,
        null=True
    )

    # ✅ Make nullable to avoid migration issues
    government_id = models.FileField(
        upload_to='ids/',
        null=True,
        blank=True
    )

    is_verified_doctor = models.BooleanField(default=False)

    verification_status = models.CharField(
        max_length=20,
        choices=VERIFICATION_STATUS,
        default='pending'
    )

    verified_at = models.DateTimeField(
        blank=True,
        null=True
    )

    verified_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="verified_doctors"
    )

    rejection_reason = models.TextField(
        blank=True,
        null=True
    )

    # ------------------
    # Admin fields
    # ------------------
    admin_code = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    can_approve_doctors = models.BooleanField(default=False)

    access_level = models.IntegerField(
        default=1,
        help_text="1 = basic admin, 2 = super admin"
    )

    def __str__(self):
        return f"{self.username} ({self.role})"

    @property
    def is_doctor_verified(self):
        return self.role == "doctor" and self.is_verified_doctor



# --------------------------
# Doctor Profile / Appointment
# --------------------------
from django.db import models
from django.conf import settings

# users/models.py
# users/models.py

DAYS_OF_WEEK = [
    ('monday', 'Monday'),
    ('tuesday', 'Tuesday'),
    ('wednesday', 'Wednesday'),
    ('thursday', 'Thursday'),
    ('friday', 'Friday'),
    ('saturday', 'Saturday'),
    ('sunday', 'Sunday'),
]

# users/models.py
class DoctorProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )
    specialization = models.CharField(max_length=100, blank=True, null=True)
    experience_years = models.PositiveIntegerField(default=0)
    bio = models.TextField(blank=True, null=True)

    # Profile & contact
    profile_image = models.ImageField(upload_to='doctor_profiles/', blank=True, null=True)
    clinic_address = models.TextField(blank=True, null=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)

    # Payment / account details
    bank_name = models.CharField(max_length=100, blank=True, null=True)
    account_number = models.CharField(max_length=50, blank=True, null=True)
    account_holder_name = models.CharField(max_length=100, blank=True, null=True)
    upi_id = models.CharField(max_length=100, blank=True, null=True)

    consultation_fee = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    verified = models.BooleanField(default=False)
    availability = models.JSONField(default=dict, blank=True)


    def __str__(self):
        return f"Dr. {self.user.username} ({self.specialization})"


from django.db import models
from django.utils import timezone

class Appointment(models.Model):
    doctor = models.ForeignKey(
        'DoctorProfile',
        on_delete=models.CASCADE,
        related_name='appointments'
    )

    patient_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        default=""
    )

    phone = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        default=""
    )

    email = models.EmailField(
        blank=True,
        null=True,
        default=""
    )

    date = models.DateField(
        default=timezone.now
    )

    time = models.TimeField(
        default=timezone.now
    )

    reason = models.TextField(
        blank=True,
        null=True,
        default=""
    )

    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("confirmed", "Confirmed"),
            ("completed", "Completed"),
            ("cancelled", "Cancelled"),
        ],
        default="pending"
    )

    created_at = models.DateTimeField(
        default=timezone.now
    )

    def __str__(self):
        return f"{self.patient_name or 'Patient'} → {self.doctor.user.username} ({self.date} {self.time})"


# --------------------------
# Skin Tracking / Reports
# --------------------------
class SkinProgress(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="skin_images/")
    detection_result = models.TextField(blank=True, null=True)
    ai_confidence = models.FloatField(default=0)
    confidence_scores_json = models.JSONField(default=dict, blank=True)

    # ✅ Keep this for tracking improvement
    improvement_score = models.FloatField(default=0)  

    # ✅ Add these to fix the /send-report error
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    annotated_image = models.ImageField(upload_to="annotated_reports/", null=True, blank=True)


class PatientReport(models.Model):
    progress = models.ForeignKey(SkinProgress, on_delete=models.CASCADE, related_name='reports')
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=16, null=True, blank=True)  # <-- add null=True
    duration_days = models.IntegerField(null=True, blank=True)
    itchiness = models.BooleanField(null=True, blank=True)
    other_notes = models.TextField(blank=True)

    multimodal_prediction = models.CharField(max_length=128, blank=True)
    multimodal_confidence = models.FloatField(null=True, blank=True)

    reviewed_by = models.ForeignKey(DoctorProfile, null=True, blank=True, on_delete=models.SET_NULL)
    review_notes = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=[
        ("pending", "Pending Review"),
        ("reviewed", "Reviewed"),
        ("followup", "Follow-Up Needed")
    ], default="pending")
    annotated_image = models.ImageField(upload_to="annotated_reports/", null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    reviewed_at = models.DateTimeField(blank=True, null=True)



class MedicalResource(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to='resources/', blank=True, null=True)
    url = models.URLField(blank=True, null=True)  # for external links
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


from django.conf import settings

class DoctorReport(models.Model):
    patient = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="sent_reports"
    )
    doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="received_reports"
    )

    detected_issues = models.JSONField()
    confidence_scores = models.JSONField()
    annotated_image = models.TextField()  # base64
    created_at = models.DateTimeField(auto_now_add=True)

    status = models.CharField(
        max_length=20,
        choices=[('pending', 'Pending'), ('reviewed', 'Reviewed')],
        default='pending'
    )

    def __str__(self):
        return f"{self.patient.username} → Dr. {self.doctor.username}"

from django.conf import settings
from django.db import models

class Review_Report_by_Doctor(models.Model):
    # patient
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='received_reviews'  # unique reverse accessor for patient
    )
    # doctor
    doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='given_reviews'  # already unique
    )
    patient_report = models.ForeignKey(
        'PatientReport',
        on_delete=models.CASCADE
    )
    content = models.TextField()
    rating = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review by Dr.{self.doctor.username} for {self.user.username}"

from django.conf import settings
from django.db import models
from django.db import models
from django.conf import settings

class MyAIReport(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('reviewed', 'Reviewed'),
        ('followup', 'Follow-Up Needed'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='my_ai_reports'
    )
    doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,       # ✅ Use SET_NULL so doctor can be null initially
        null=True,                        # ✅ Allow null
        blank=True,                       # ✅ Allow blank in forms/admin
        related_name='my_ai_reports_given'
    )
    detected_issues = models.JSONField(default=list, blank=True)
    confidence_scores = models.IntegerField(default=0)
    confidence_scores_json = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    annotated_image = models.ImageField(upload_to='ai_reports/annotated_images/', null=True, blank=True)
    skin_progress = models.ForeignKey(
        'SkinProgress',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='ai_reports'
    )
    prediction = models.CharField(max_length=255, blank=True)

    # ✅ NEW FIELDS
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    review_notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        if self.doctor:
            return f"{self.user.username} → Dr. {self.doctor.username} ({self.status})"
        return f"{self.user.username} → No doctor assigned ({self.status})"


class Notification(models.Model):
    NOTIFICATION_TYPES = (
        ("report_sent", "AI Report Sent"),
        ("report_reviewed", "AI Report Reviewed"),
    )

    recipient = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notifications"
    )
    message = models.TextField()
    notification_type = models.CharField(max_length=50, choices=NOTIFICATION_TYPES)
    related_report = models.ForeignKey(
        "MyAIReport",
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    url = models.CharField(max_length=255, blank=True, null=True)  # <-- Add this
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Notification → {self.recipient.username}"
