# users/utils.py

import json
from django.utils import timezone
from django.core.files.base import ContentFile
from users.models import SkinProgress, MyAIReport
from datetime import datetime

# -------------------------------
# Save prediction result
# -------------------------------
def save_prediction_result(user, detected_issues, confidence_scores, image=None):
    """
    Save AI prediction result as SkinProgress for the user.
    """
    avg_confidence = int(sum(confidence_scores.values())/len(confidence_scores)) if confidence_scores else 0
    detection_result = ", ".join(detected_issues)

    skin_progress = SkinProgress.objects.create(
        user=user,
        detection_result=detection_result,
        confidence_scores_json=confidence_scores,
        ai_confidence=avg_confidence,
        age=getattr(user, 'age', None),
        gender=getattr(user, 'gender', None),
        annotated_image=image
    )
    return skin_progress


# -------------------------------
# Get baseline progress
# -------------------------------
def get_baseline_progress(user):
    """
    Return the earliest SkinProgress for the user to compare AI progress.
    """
    return SkinProgress.objects.filter(user=user).order_by("created_at").first()


# -------------------------------
# Analyze progress
# -------------------------------
def analyze_progress(current_scores, baseline_progress):
    """
    Compare current confidence scores with baseline to generate a summary.
    """
    summary = {}
    baseline_scores = baseline_progress.confidence_scores_json or {}

    for condition, score in current_scores.items():
        base_score = baseline_scores.get(condition, 0)
        change = score - base_score
        summary[condition] = {
            "baseline": base_score,
            "current": score,
            "change": change
        }

    return {
        "status": "Analysis complete.",
        "summary": summary
    }


# -------------------------------
# Predict symptom diseases
# -------------------------------
def predict_symptom_diseases(symptoms, top_k=5, min_prob=0.3):
    """
    Dummy symptom-to-disease prediction.
    Replace with real ML logic.
    Returns list of tuples: [(disease_name, probability)]
    """
    # Example static mapping for demo
    example_mapping = {
        "dark circles": 0.78,
        "freckles": 0.74,
        "wrinkles": 0.65
    }
    results = [(disease, prob) for disease, prob in example_mapping.items() if prob >= min_prob]
    return results[:top_k]


# -------------------------------
# Get follow-up questions
# -------------------------------
def get_follow_up_questions(issues):
    """
    Return follow-up questions based on detected issues.
    """
    questions = []
    for issue in issues:
        questions.append({
            "condition": issue,
            "question": f"Do you have additional symptoms for {issue}?"
        })
    return questions
