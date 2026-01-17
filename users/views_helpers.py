# users/views_helpers.py
import base64
from django.core.files.base import ContentFile
from .models import DoctorProfile, DoctorReport, MyAIReport

def send_report_to_doctor_internal(user, doctor_id, skin_progress):
    """
    Create DoctorReport & MyAIReport simultaneously for a given user, doctor, and SkinProgress.
    """
    doctor_profile = DoctorProfile.objects.get(
        id=doctor_id,
        user__is_verified_doctor=True
    )

    # ---- DoctorReport (doctor inbox) ----
    annotated_base64 = ""
    if skin_progress.annotated_image:
        with skin_progress.annotated_image.open("rb") as f:
            annotated_base64 = base64.b64encode(f.read()).decode("utf-8")

    DoctorReport.objects.create(
        patient=user,
        doctor=doctor_profile.user,
        detected_issues=skin_progress.detection_result.split(", ") if skin_progress.detection_result else [],
        confidence_scores=skin_progress.confidence_scores_json or {},
        annotated_image=annotated_base64,
        status="pending"
    )

    # ---- MyAIReport (user dashboard) ----
    avg_conf = int(sum(skin_progress.confidence_scores_json.values()) / len(skin_progress.confidence_scores_json)) if skin_progress.confidence_scores_json else 0
    prediction = skin_progress.detection_result.split(",")[0] if skin_progress.detection_result else "Not available"

    age = getattr(skin_progress, 'age', None) or "—"
    gender = getattr(skin_progress, 'gender', None) or "—"
    annotated_file = skin_progress.annotated_image if skin_progress.annotated_image else None

    MyAIReport.objects.create(
        user=user,
        doctor=doctor_profile.user,
        skin_progress=skin_progress,
        age=age,
        gender=gender,
        detected_issues=skin_progress.detection_result.split(", ") if skin_progress.detection_result else [],
        confidence_scores=skin_progress.confidence_scores_json or {},
        confidence_scores_avg=avg_conf,
        prediction=prediction,
        annotated_image=annotated_file,
        status="pending"
    )
