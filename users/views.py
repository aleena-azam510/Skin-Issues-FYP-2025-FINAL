# users/views.py
import base64
from django.utils import timezone

import json
import uuid
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate, get_user_model
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse, JsonResponse
from django.utils.http import url_has_allowed_host_and_scheme
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.urls import reverse
from django.template.loader import render_to_string
from django.conf import settings
from django.core.mail import send_mail
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import logging
from predictor.views import predict_view
from reviews.models import Review
from users.models import CustomUser, DoctorProfile, PatientReport, SkinProgress
from users.services.dashboard_stats import get_skin_stats
from .forms import AppointmentForm, CustomUserCreationForm, DoctorProfileForm, SkinProgressForm
from django.contrib.auth.backends import ModelBackend # Necessary for explicit backend
from .forms import CustomUserCreationForm
from django.http import JsonResponse
logger = logging.getLogger(__name__)
User = get_user_model()
# users/views.py
from django.views.decorators.http import require_POST


from django.utils import timezone
from datetime import timedelta

from django.core.files.base import ContentFile  # <-- add this

# --------------------------
# Role-based redirection
# --------------------------


# ----------------------------------------------------
# SOCIAL AUTH REDIRECT/ROLE CHECK (django-allauth flow)
# ----------------------------------------------------
# ----------------------------------------------------
# SOCIAL AUTH REDIRECT / ROLE CHECK (django-allauth)
# ----------------------------------------------------
@login_required
def social_auth_complete(request):
    """
    Called after successful social sign-up/sign-in via allauth.
    If the user's custom 'role' field is not set, redirect safely.
    """

    user = request.user
    role = getattr(user, "role", None)

    # Safety fallback for social-auth users without role
    if not role:
        messages.warning(
            request,
            "Please complete your profile before continuing."
        )
        return redirect("auth_page")  # or profile setup page

    return redirect_by_role(user)


# --------------------------
# AUTH PAGE (SIGNUP + LOGIN)
# --------------------------
import logging
logger = logging.getLogger(__name__)

def auth_view(request):
    logger.info("🔹 Entered auth_view — method=%s", request.method)

    registration_form = CustomUserCreationForm()
    login_form = AuthenticationForm()
    active_panel = ""

    # Already logged-in users
    if request.user.is_authenticated:
        logger.info(
            "⚠️ User already authenticated: %s (role=%s)",
            request.user.username,
            getattr(request.user, "role", "N/A"),
        )
        return redirect_by_role(request.user)

    if request.method == "POST":
        logger.debug("📩 POST data received: %s", request.POST.dict())

        # ------------------------
        # SIGNUP
        # ------------------------
        if "signup_submit" in request.POST:
            logger.info("🧾 Signup form submitted")
            active_panel = "right-panel-active"
            registration_form = CustomUserCreationForm(
                request.POST, request.FILES
            )

            if registration_form.is_valid():
                user = registration_form.save(commit=False)
                role = user.role

                logger.info(
                    "✅ Registration valid — user=%s role=%s",
                    user.username,
                    role,
                )

                try:
                    if role == "user":
                        user.is_active = True
                        user.save()
                        login(
                            request,
                            user,
                            backend="django.contrib.auth.backends.ModelBackend",
                        )
                        return redirect_by_role(user)

                    elif role == "doctor":
                        user.is_active = True                 # Allow login
                        user.is_verified_doctor = False       # Lock features
                        user.save()

                        login(
                            request,
                            user,
                            backend="django.contrib.auth.backends.ModelBackend",
                        )

                        messages.info(
                            request,
                            "Your account has been created. Some features will be unlocked after admin verification."
                        )

                        return redirect("doctor_home")         # ✅ Correct target


                    elif role == "clerk":
                        user.is_active = False
                        user.save()

                        token = default_token_generator.make_token(user)
                        uid = urlsafe_base64_encode(force_bytes(user.pk))
                        verification_link = request.build_absolute_uri(
                            reverse(
                                "verify_email",
                                kwargs={
                                    "uidb64": uid,
                                    "token": token,
                                },
                            )
                        )

                        send_mail(
                            subject="Activate Your Clerk Account",
                            message=(
                                f"Hello {user.username},\n\n"
                                f"Verify your account:\n{verification_link}"
                            ),
                            from_email=None,
                            recipient_list=[user.email],
                        )

                        messages.success(
                            request,
                            "Clerk account created. Check your email to activate.",
                        )
                        return redirect("auth_page")

                    elif role == "admin":
                        user.is_active = False
                        user.save()
                        messages.success(
                            request,
                            "Admin account created. Awaiting verification.",
                        )
                        return redirect("auth_page")

                    else:
                        logger.error("❌ Unknown role: %s", role)
                        messages.error(request, "Invalid role selected.")

                except Exception as e:
                    logger.exception("💥 Exception during signup")
                    messages.error(
                        request,
                        "Unexpected error occurred during signup.",
                    )

            else:
                logger.warning(
                    "❌ Registration form invalid: %s",
                    registration_form.errors,
                )
                messages.error(
                    request,
                    "Registration failed. Please fix the errors.",
                )

        # ------------------------
        # SIGNIN
        # ------------------------
        elif "signin_submit" in request.POST:
            logger.info("🔐 Signin form submitted")
            login_form = AuthenticationForm(
                request, data=request.POST
            )

            if login_form.is_valid():
                user = login_form.get_user()

                if user and user.is_active:
                    login(
                        request,
                        user,
                        backend="django.contrib.auth.backends.ModelBackend",
                    )
                    return redirect_by_role(user)

                messages.error(
                    request,
                    "Account inactive. Check your email.",
                )

            else:
                logger.warning(
                    "❌ Login form invalid: %s",
                    login_form.errors,
                )
                messages.error(
                    request,
                    "Invalid username or password.",
                )

    return render(
        request,
        "users/register.html",
        {
            "registration_form": registration_form,
            "login_form": login_form,
            "active_panel": active_panel
            or (
                "right-panel-active"
                if registration_form.errors
                else ""
            ),
        },
    )

from django.contrib import messages
from django.shortcuts import redirect

def doctor_verified_required(view_func):
    def wrapper(request, *args, **kwargs):
        if (
            request.user.role == "doctor"
            and not request.user.is_verified_doctor
        ):
            messages.warning(
                request,
                "Your account is pending verification. This feature is locked."
            )
            return redirect("doctor_home")
        return view_func(request, *args, **kwargs)
    return wrapper

# --------------------------
# EMAIL VERIFICATION
# --------------------------
def verify_email(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        login(
            request,
            user,
            backend="django.contrib.auth.backends.ModelBackend",
        )
        messages.success(request, "Your account has been activated!")
        return redirect_by_role(user)

    messages.error(request, "Invalid or expired activation link.")
    return redirect("auth_page")





# ... (rest of your views: logout_view, index_page, doctor_home, etc.)

# --------------------------
# LOGOUT
# --------------------------
def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('auth_page')


# --------------------------
# HOMEPAGE / INDEX
# --------------------------
@login_required
def index_page(request):
    reviews = Review.objects.all().order_by('-created_at')
    return render(request, 'users/index.html', {'reviews': reviews, 'total_reviews_count': reviews.count()})


# --------------------------
# DOCTOR DASHBOARD
# --------------------------
@login_required
def doctor_home(request):
    return render(request, 'users/doctor_home.html')


# --------------------------
# ADMIN DASHBOARD
# --------------------------
@login_required
@user_passes_test(lambda u: u.role == 'admin')
def admin_dashboard(request):
    unverified_doctors = CustomUser.objects.filter(
        role='doctor',
        is_verified_doctor=False
    )

    verified_doctors = CustomUser.objects.filter(
        role='doctor',
        is_verified_doctor=True
    ).count()

    total_users = CustomUser.objects.count()

    return render(request, 'users/admin_dashboard.html', {
        'unverified_doctors': unverified_doctors,
        'verified_doctors': verified_doctors,
        'total_users': total_users,
    })


def is_admin(user):
    return user.is_staff or user.is_superuser


@user_passes_test(is_admin)
def verify_doctor(request, doctor_id):
    doctor = get_object_or_404(CustomUser, id=doctor_id, role='doctor')

    if request.method == "POST":
        action = request.POST.get('action')
        reason = request.POST.get('rejection_reason', '')

        if action == 'approve':
            doctor.is_verified_doctor = True
            doctor.rejection_reason = ''
            doctor.verified_at = timezone.now()
            doctor.save()

            # Send email notification
            send_mail(
                'Your Account is Verified',
                f'Hello Dr. {doctor.username},\n\nYour account has been approved. You now have full access.',
                'admin@healthcare.com',
                [doctor.email],
                fail_silently=True,
            )
            messages.success(request, f"{doctor.username} approved successfully.")

        elif action == 'reject':
            doctor.is_verified_doctor = False
            doctor.rejection_reason = reason
            doctor.save()

            # Send email notification
            send_mail(
                'Your Account Verification was Rejected',
                f'Hello Dr. {doctor.username},\n\nYour account verification was rejected for the following reason:\n"{reason}"\nPlease contact admin for corrections.',
                'admin@healthcare.com',
                [doctor.email],
                fail_silently=True,
            )
            messages.error(request, f"{doctor.username} rejected successfully.")

        return redirect('admin_dashboard')


# --------------------------
# INDEX / HOME PAGE
# --------------------------
# or your actual model

def index_page(request):
    # Fetch all reviews
    reviews = Review.objects.all().order_by('-created_at')

    # Fetch unread notifications for authenticated users
    unread_notifications = []
    if request.user.is_authenticated:
        unread_notifications = request.user.notifications.filter(is_read=False)

    context = {
        'reviews': reviews,
        'total_reviews_count': reviews.count(),
        'unread_notifications': unread_notifications,
    }
    return render(request, 'users/index.html', context)

# --------------------------
# LOGOUT
# --------------------------
def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('auth_page')


# --------------------------
# PDF DOWNLOAD (ADMIN ONLY)
# --------------------------
@user_passes_test(lambda u: u.is_superuser)
def download_users_pdf(request):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 750, "Registered Users List")
    users = User.objects.all().order_by('username')
    y = 700
    for user in users:
        p.setFont("Helvetica", 12)
        p.drawString(50, y, user.username)
        p.drawString(200, y, user.email or "N/A")
        p.drawString(400, y, user.role)
        y -= 20
    p.showPage()
    p.save()
    buffer.seek(0)
    return HttpResponse(buffer, content_type='application/pdf')


@login_required
def developer_team_view(request):
    return render(request, 'developer_Team.html')


from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from users.models import CustomUser

@login_required
def model_page_view(request):
    # Only fetch **verified doctors**
    verified_doctors = DoctorProfile.objects.filter(
        user__role="doctor",
        user__is_verified_doctor=True
    ).select_related('user')  # so we can access doctor.user.username

    context = {
        'is_authenticated': True,
        'verified_doctors': verified_doctors
    }

    return render(request, "modelPage.html", context)



@login_required
def article_view(request):
    return render(request, 'article.html', {'is_authenticated': True})

# --------------------------
# USER DASHBOARD (Skin Progress Tracker)
# --------------------------

# users/views.py
@login_required
def user_dashboard_home(request):
        

    reports = MyAIReport.objects.filter(user=request.user)

    context = {
        "reports": reports,
        "reports_count": reports.count(),
        "pending_count": reports.filter(status="pending").count(),
        "reviewed_count": reports.filter(status="reviewed").count(),
    }


    return render(request, "dashboard/dashboard_home.html", context)

# users/views.py
from django.contrib.auth.decorators import login_required
from reviews.models import Review



@login_required
def skin_tracker(request):
    if request.method == "POST" and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        image = request.FILES.get("image")
        if not image:
            return JsonResponse({"status": "error", "message": "No image uploaded"}, status=400)

        try:
            # ----------------------
            # 1️⃣ Save SkinProgress
            # ----------------------
            skin_progress = SkinProgress.objects.create(
                user=request.user,
                image=image,
                detection_result="",
                ai_confidence=0
            )

            # ----------------------
            # 2️⃣ Run AI prediction
            # ----------------------
            request.FILES["file"] = image
            response = predict_view(request)
            if response.status_code != 200:
                skin_progress.delete()
                return JsonResponse({"status": "error", "message": "AI analysis failed"}, status=500)

            result = json.loads(response.content)
            detected_issues = result.get('detected_issues', [])
            confidence_scores = result.get('confidence_scores', {})
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0

            # ----------------------
            # 3️⃣ Save annotated image if returned
            # ----------------------
            img_base64 = result.get("annotated_image")
            if img_base64:
                skin_progress.annotated_image.save(
                    f"{uuid.uuid4()}.jpg",
                    ContentFile(base64.b64decode(img_base64))
                )

            # ----------------------
            # 4️⃣ Update SkinProgress
            # ----------------------
            skin_progress.detection_result = ", ".join(detected_issues)
            skin_progress.ai_confidence = avg_confidence
            skin_progress.save()

            # ----------------------
            # 5️⃣ Send reports to doctor & MyAIReport
            # ----------------------
            doctor_id = request.POST.get("doctor_id")  # if selected on frontend
            if doctor_id:
                from .views_helpers import send_report_to_doctor_internal
                send_report_to_doctor_internal(request.user, doctor_id, skin_progress)

            # ----------------------
            # 6️⃣ Prepare response
            # ----------------------
            progress_summary = result.get('progress_summary', {})
            improvement = progress_summary.get('overall_improvement', 0) if isinstance(progress_summary, dict) else 0

            return JsonResponse({
                "status": "success",
                "data": {
                    "confidence": round(avg_confidence, 2),
                    "conditions": detected_issues,
                    "image_url": skin_progress.image.url,
                    "date": skin_progress.created_at.strftime("%b %d, %Y"),
                    "improvement": improvement,
                }
            })

        except Exception as e:
            logger.exception("Skin tracker failed")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    # GET request
    context = get_skin_stats(request.user)
    return render(request, "dashboard/skin_tracker.html", context)


# In delete_skin_image function
# views.py - Simple delete view that always returns JSON

@login_required
@require_POST
def delete_skin_image(request, id):
    """Debug delete view"""
    print(f"DELETE REQUEST RECEIVED for ID: {id}")
    print(f"User: {request.user}")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Body: {request.body}")
    
    try:
        # Try to get the object
        obj = SkinProgress.objects.get(id=id, user=request.user)
        print(f"Found object: {obj}")
        
        # Delete it
        obj.delete()
        print("Object deleted successfully")
        
        # Return success response
        return JsonResponse({
            "success": True,
            "message": "Deleted successfully",
            "deleted_id": id
        }, json_dumps_params={'indent': 2})
        
    except SkinProgress.DoesNotExist:
        print("Object not found")
        return JsonResponse({
            "success": False,
            "error": "Image not found or you don't have permission"
        }, status=404, json_dumps_params={'indent': 2})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500, json_dumps_params={'indent': 2})
@login_required
@require_POST
def delete_all_skin_history(request):
    print("DELETE ALL HISTORY REQUEST RECEIVED")
    print("User:", request.user)

    deleted_count, _ = SkinProgress.objects.filter(
        user=request.user
    ).delete()

    return JsonResponse({
        "success": True,
        "deleted": deleted_count
    })
@login_required
def uv_protection(request):
    return render(request, "dashboard/uv_protection.html")


@login_required
def lifestyle(request):
    return render(request, "dashboard/lifestyle.html")


@login_required
def sleep_stress(request):
    return render(request, "dashboard/sleep.html")


@login_required
def habits(request):
    return render(request, "dashboard/habits.html")


@login_required
def journal(request):
    return render(request, "dashboard/journal.html")


# users/views.py - Corrected redirect_by_role
# users/views.py
def redirect_by_role(user):
    role = getattr(user, 'role', 'user')

    if role == 'doctor':
        return redirect('doctor_home')   # ✅ FIXED
    elif role == 'admin':
        return redirect('admin_dashboard')
    elif role == 'user':
        return redirect('user_dashboard_home')
    else:
        return redirect('index_page')

    # users/views.py
@login_required
def social_auth_complete(request):
    """
    Called after successful social sign-up/sign-in via allauth.
    ...
    """
    return redirect_by_role(request.user)

from django.contrib.auth.decorators import login_required

@login_required
def dashboard_redirect(request):
    return redirect_by_role(request.user)

from django.shortcuts import render
from .models import DoctorProfile, DoctorReport, MedicalResource, MyAIReport, Notification
from django.db.models import Avg

def doctors_list(request):
    # Only verified doctors
    doctors = DoctorProfile.objects.filter(
        user__role='doctor',
        user__is_verified_doctor=True
    )

    # Get distinct specializations
    specializations = doctors.values_list('specialization', flat=True).distinct()

    # Average experience (optional)
    avg_experience = int(doctors.aggregate(avg_exp=Avg('experience_years'))['avg_exp'] or 0)

    context = {
        'doctors': doctors,
        'specializations': specializations,  # <-- pass this to template
        'avg_experience': avg_experience,
    }
    return render(request, 'users/doctors_list.html', context)


def public_doctor_profile(request, doctor_id):
    doctor = get_object_or_404(DoctorProfile, id=doctor_id, verified=True)
    return render(request, "users/public_doctor_profile.html", {"doctor": doctor})
@login_required
def book_appointment(request, doctor_id):
    doctor = get_object_or_404(DoctorProfile, id=doctor_id, verified=True)

    if request.method == "POST":
        form = AppointmentForm(request.POST)
        if form.is_valid():
            appointment = form.save(commit=False)
            appointment.user = request.user
            appointment.doctor = doctor
            appointment.save()
            messages.success(request, "Appointment booked successfully.")
            return redirect('user_appointments')
    else:
        form = AppointmentForm()

    return render(request, "users/book_appointment.html", {"form": form, "doctor": doctor})


from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import DoctorProfile, PatientReport

@login_required
def doctor_reports(request):
    doctor_profile = request.user.doctorprofile

    reports = PatientReport.objects.filter(
        status="pending",
        reviewed_by__isnull=True
    ).select_related(
        "progress",
        "progress__user"
    ).order_by("-created_at")

    return render(request, "users/doctor_reports.html", {
        "reports": reports
    })


from django.shortcuts import render, get_object_or_404, redirect
from .models import MyAIReport  # or DoctorReport if that’s your model

@login_required
def review_report(request, report_id):
    report = get_object_or_404(MyAIReport, id=report_id)

    if request.method == "POST":
        report.review_notes = request.POST.get("review_notes", "").strip()
        report.status = request.POST.get("status", "reviewed")

        if report.status == "reviewed":
            report.updated_at = timezone.now()

        report.save()
        messages.success(request, "Review saved successfully.")
        return redirect("doctor_pending_reports")

    # Filter detected issues to remove numbers
    if report.detected_issues:
        filtered_issues = [i for i in report.detected_issues if not str(i).isdigit()]
    else:
        filtered_issues = []

    context = {
        "report": report,
        "patient": report.user,
        "progress": report.skin_progress,
        "filtered_issues": filtered_issues,
    }

    return render(request, "users/review_report.html", context)

@login_required
def my_reports(request):
    reports = PatientReport.objects.filter(progress__user=request.user).order_by('-created_at')
    return render(request, "users/my_reports.html", {"reports": reports})


def verified_doctors_list(request):
    doctors = DoctorProfile.objects.filter(
        user__role="doctor",
        user__is_verified_doctor=True
    )

    specializations = doctors.values_list(
        "specialization",
        flat=True
    ).distinct()

    return render(
        request,
        "users/doctors_list.html",
        {
            "doctors": doctors,
            "specializations": specializations,
        }
    )



def doctor_profile(request, doctor_id):
    doctor = get_object_or_404(DoctorProfile, id=doctor_id, user__is_verified_doctor=True)
    return render(request, 'users/doctor_profile.html', {'doctor': doctor})

@login_required
def doctor_profile_view(request):
    if request.user.role != "doctor":
        messages.error(request, "Access denied")
        return redirect("index_page")

    # Try to get existing profile
    profile, created = DoctorProfile.objects.get_or_create(user=request.user)

    if request.method == "POST":
        form = DoctorProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Your public profile has been saved.")
            return redirect("doctor_home")
    else:
        form = DoctorProfileForm(instance=profile)

    return render(request, "users/doctor_profile.html", {"form": form, "profile": profile})


from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import DoctorProfile, Appointment

def book_appointment(request, doctor_id):
    doctor = get_object_or_404(DoctorProfile, pk=doctor_id)

    if request.method == "POST":
        date = request.POST.get("date")
        time = request.POST.get("time")
        patient_name = request.POST.get("patient_name")
        phone = request.POST.get("phone")
        email = request.POST.get("email")
        reason = request.POST.get("reason")

        if not all([date, time, patient_name, phone]):
            messages.error(request, "Please fill all required fields.")
            return redirect("book_appointment", doctor_id=doctor.id)

        Appointment.objects.create(
            doctor=doctor,
            patient_name=patient_name,
            phone=phone,
            email=email,
            date=date,
            time=time,
            reason=reason,
        )

        messages.success(request, "Appointment successfully booked!")
        return redirect("public_doctor_profile", doctor_id=doctor.id)

    return render(
        request,
        "users/book_appointments.html",
        {"doctor": doctor},
    )


def start_video_consultation(request, doctor_id):
    doctor = get_object_or_404(DoctorProfile, pk=doctor_id)
    # Redirect to a dummy video consultation page
    return render(request, 'users/video_consultation.html', {'doctor': doctor})

def medical_articles(request, doctor_id):
    doctor = get_object_or_404(DoctorProfile, pk=doctor_id)
    # Load articles (dummy)
    articles = [
        {"title": "Healthy Skin Tips", "url": "#"},
        {"title": "Managing Stress", "url": "#"}
    ]
    return render(request, 'users/medical_articles.html', {'doctor': doctor, 'articles': articles})

def share_profile(request, doctor_id):
    doctor = get_object_or_404(DoctorProfile, pk=doctor_id)
    share_url = request.build_absolute_uri(doctor.get_absolute_url())
    messages.success(request, f"Share this profile link: {share_url}")
    return redirect('public_doctor_profile', doctor_id=doctor.id)



@login_required
def doctor_prescriptions(request):
    if request.user.role != "doctor":
        return redirect("user_dashboard_home")

    return render(request, "users/doctor_prescriptions.html")

# users/views.py
from django.shortcuts import render, get_object_or_404, redirect
from .models import DoctorProfile, DAYS_OF_WEEK
from .forms import DoctorProfileForm

def doctor_profile_manage(request):
    profile, created = DoctorProfile.objects.get_or_create(user=request.user)

    if request.method == "POST":
        form = DoctorProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            profile = form.save(commit=False)
            # Save availability
            availability_data = {}
            for day, _ in DAYS_OF_WEEK:
                field_name = f"{day}_times"
                availability_data[day] = form.cleaned_data.get(field_name, "")
            profile.availability = availability_data
            profile.save()
            return redirect("doctor_profile")  # or some success page
    else:
        form = DoctorProfileForm(instance=profile)
        # Pre-fill availability in the form fields
        for day, _ in DAYS_OF_WEEK:
            form.fields[f"{day}_times"].initial = profile.availability.get(day, "")

    return render(request, "users/doctor_profile_form.html", {
        "form": form,
        "availability_fields": [form[f"{day}_times"] for day, _ in DAYS_OF_WEEK],
        "profile": profile,
    })
    
def public_doctor_profile(request, doctor_id):
    doctor = get_object_or_404(
        DoctorProfile,
        id=doctor_id,
        user__role="doctor",
        user__is_verified_doctor=True
    )

    return render(
        request,
        "users/public_doctor_profile.html",
        {"doctor": doctor}
    )
    
    
@login_required
def doctor_appointments(request):
    if request.user.role != "doctor":
        messages.error(request, "Access denied")
        return redirect("index_page")

    doctor_profile = get_object_or_404(DoctorProfile, user=request.user)
    appointments = Appointment.objects.filter(doctor=doctor_profile)

    # 🔢 Statistics
    total_patients = appointments.values("patient_name").distinct().count()

    today = timezone.now().date()
    todays_appointments = appointments.filter(date=today).count()  # <-- fixed
    week_end = today + timedelta(days=7)
    upcoming_week = appointments.filter(date__range=[today, week_end]).count()  # <-- fixed

    context = {
        "appointments": appointments.order_by("date", "time"),
        "total_patients": total_patients,
        "todays_appointments": todays_appointments,
        "upcoming_week": upcoming_week,
    }

    return render(request, "users/doctor_appointments.html", context)

# @login_required
# def medical_resources(request):
#     if request.user.role != "doctor":
#         messages.error(request, "Access denied")
#         return redirect('index_page')

#     resources = MedicalResource.objects.all().order_by('-created_at')
#     return render(request, "users/medical_resources.html", {"resources": resources})



from django.utils import timezone

def doctor_pending_reports(request):
    reports = MyAIReport.objects.filter(status='pending').order_by('-created_at')
    now = timezone.now()
    # Annotate each report with age in days
    for r in reports:
        r.age_days = (now - r.created_at).days
    context = {
        'reports': reports,
    }
    return render(request, 'users/doctor_reports.html', context)


from .models import Review_Report_by_Doctor  # <- Correct model

@login_required
def user_reviews(request):
    reviews = MyAIReport.objects.filter(
        user=request.user,
        status="reviewed"
    ).select_related("doctor").order_by("-updated_at")

    total_reviews_count = reviews.count()
    unique_doctors = reviews.values("doctor").distinct().count()
    recent_reviews_count = reviews.filter(
        updated_at__gte=timezone.now() - timezone.timedelta(days=30)
    ).count()

    # Filter out numeric keys from confidence_scores_json
    for report in reviews:
        if report.confidence_scores_json:
            report.confidence_scores_json = {
                k: v for k, v in report.confidence_scores_json.items() if not k.isdigit()
            }

    context = {
        "reviews": reviews,
        "total_reviews_count": total_reviews_count,
        "unique_doctors": unique_doctors,
        "recent_reviews_count": recent_reviews_count,
        "average_rating": None,
    }

    return render(request, "dashboard/reviews.html", context)

# -----------------------------
# SEND REPORT TO DOCTOR
# -----------------------------


from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.urls import reverse
from users.models import (
    SkinProgress,
    PatientReport,
    DoctorReport,
    MyAIReport,
    DoctorProfile,
    Notification
)
import base64

@login_required
def send_report_to_doctor(request):
    if request.method != "POST" or request.headers.get("X-Requested-With") != "XMLHttpRequest":
        return JsonResponse({"status": "error", "message": "Invalid request"}, status=400)

    doctor_id = request.POST.get("doctor_id")
    if not doctor_id:
        return JsonResponse({"status": "error", "message": "No doctor selected"}, status=400)

    # Validate doctor
    try:
        doctor_profile = DoctorProfile.objects.get(id=doctor_id, user__is_verified_doctor=True)
    except DoctorProfile.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Doctor not found or not verified"}, status=404)

    # Get latest AI report without assigned doctor
    ai_report = MyAIReport.objects.filter(user=request.user, doctor__isnull=True).order_by('-created_at').first()
    if not ai_report:
        return JsonResponse({"status": "error", "message": "No AI report available to send"}, status=404)

    # Assign doctor and update status
    ai_report.doctor = doctor_profile.user
    ai_report.status = "pending"
    ai_report.save()

    # Create PatientReport
    latest_progress = ai_report.skin_progress
    patient_report = PatientReport.objects.create(
        progress=latest_progress,
        age=latest_progress.age,
        gender=latest_progress.gender,
        multimodal_prediction=latest_progress.detection_result or "",
        multimodal_confidence=latest_progress.ai_confidence or 0,
        reviewed_by=doctor_profile,
        status="pending"
    )

    # Encode annotated image if exists
    annotated_base64 = ""
    if latest_progress and latest_progress.annotated_image:
        with latest_progress.annotated_image.open("rb") as f:
            annotated_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Create DoctorReport
    doctor_report = DoctorReport.objects.create(
        patient=request.user,
        doctor=doctor_profile.user,
        detected_issues=latest_progress.detection_result.split(", ") if latest_progress.detection_result else [],
        confidence_scores=latest_progress.confidence_scores_json or {},
        annotated_image=annotated_base64,
        status="pending"
    )

    # ------------------------------
    # 🔔 Create notifications
    # ------------------------------

    # Doctor notification → link to review page
    Notification.objects.create(
        recipient=doctor_profile.user,
        message=f"{request.user.get_full_name() or request.user.username} sent you a new AI report.",
        notification_type="report_sent",
        related_report=ai_report,
        url=reverse("review_report", args=[doctor_report.id])  # correct doctor review URL
    )

    # User notification → link to reviews page
    Notification.objects.create(
        recipient=request.user,
        message=f"Your AI report has been sent to Dr. {doctor_profile.user.get_full_name() or doctor_profile.user.username}.",
        notification_type="report_sent",
        related_report=ai_report,
        url=reverse("reviews")  # correct user reviews URL
    )

    return JsonResponse({
        "status": "success",
        "message": f"Report sent to Dr. {doctor_profile.user.get_full_name() or doctor_profile.user.username} successfully!"
    })


# -----------------------------
# USER DASHBOARD: MY AI REPORTS


@login_required
def my_ai_reports(request):
    reports = MyAIReport.objects.filter(user=request.user).order_by('-created_at')

    pending_count = reports.filter(status='pending').count()
    reviewed_count = reports.filter(status='reviewed').count()
    total_doctors = reports.values('doctor').distinct().count()

    for report in reports:
        # Age & gender display
        report.age_display = report.skin_progress.age if report.skin_progress else getattr(report, 'age', '—')
        report.gender_display = report.skin_progress.gender if report.skin_progress else getattr(report, 'gender', '—')

        # Detected issues: remove numeric entries
        report.filtered_issues = [i for i in report.detected_issues if not str(i).isdigit()] if report.detected_issues else []

        # **Fix AI Prediction**
        # If stored as comma-separated string, split properly
        if report.prediction:
            if isinstance(report.prediction, str):
                # Convert string like "eye bags, dark circles, freckles" to list
                pred_list = [p.strip() for p in report.prediction.split(",") if p.strip() and not p.strip().isdigit()]
                report.filtered_prediction = pred_list
            elif isinstance(report.prediction, list):
                report.filtered_prediction = [p for p in report.prediction if not str(p).isdigit()]
            else:
                report.filtered_prediction = []
        else:
            report.filtered_prediction = []

        report.confidence_display = report.confidence_scores or 0

    context = {
        "reports": reports,
        "pending_count": pending_count,
        "reviewed_count": reviewed_count,
        "total_doctors": total_doctors,
    }
    return render(request, "dashboard/my_ai_reports.html", context)

# --------------------------
# Delete AI Report
# --------------------------
@login_required
def delete_ai_report(request, report_id):
    report = get_object_or_404(MyAIReport, id=report_id, user=request.user)
    if request.method == "POST":
        report.delete()
        messages.success(request, "AI report deleted successfully.")
    else:
        messages.error(request, "Invalid request.")
    return redirect("my_ai_reports")


@login_required
def delete_report(request, report_id):
    report = get_object_or_404(MyAIReport, id=report_id)
    if request.method == "POST":
        report.delete()
        messages.success(request, "Report deleted successfully!")
        return redirect('doctor_reports')
    return redirect('doctor_reports')

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import Notification
from django.utils import timezone
from django.urls import reverse

@login_required
def notifications_view(request):
    notifications = request.user.notifications.all().order_by('-created_at')
    unread_count = notifications.filter(is_read=False).count()
    recent_count = notifications.filter(
        created_at__gte=timezone.now()-timezone.timedelta(days=7)
    ).count()

    # Add URL for each notification
    for n in notifications:
        if n.related_report:
            if request.user.role == "doctor":
                # Doctor → review_report page
                n.url = reverse('review_report', args=[n.related_report.id])
            else:
                # User → reviews page
                n.url = reverse('reviews')
        else:
            n.url = None

    context = {
        'notifications': notifications,
        'unread_count': unread_count,
        'recent_count': recent_count,
    }
    return render(request, 'dashboard/notifications.html', context)


@login_required
def mark_notification_read(request, notification_id):
    """
    Mark a single notification as read.
    """
    notification = get_object_or_404(
        Notification,
        id=notification_id,
        recipient=request.user
    )
    notification.is_read = True
    notification.save()

    if request.is_ajax():
        return JsonResponse({'success': True})
    return redirect('notifications_view')


@login_required
def mark_all_notifications_read(request):
    """
    Mark all notifications of the user as read.
    """
    request.user.notifications.filter(is_read=False).update(is_read=True)
    if request.is_ajax():
        return JsonResponse({'success': True})
    return redirect('notifications_view')

@login_required
def clear_all_notifications(request):
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        # Delete all notifications for the logged-in user
        Notification.objects.filter(recipient=request.user).delete()
        return JsonResponse({"success": True, "message": "All notifications cleared"})
    return JsonResponse({"success": False, "message": "Invalid request"}, status=400)