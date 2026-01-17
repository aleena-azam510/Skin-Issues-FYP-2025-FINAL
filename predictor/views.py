import base64
from datetime import timezone
import cv2
from django.shortcuts import get_object_or_404, render
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from django.http import HttpResponse, JsonResponse
import logging
import traceback
import io
import hashlib
import joblib
import numpy as np
import pandas as pd
import json
import os 
import sys 

# --- EXISTING IMPORTS ---
from ultralytics import YOLO
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from users.models import DoctorReport, SkinProgress
from .models import SkinCondition,SkinCondition_page, User
# --- END EXISTING IMPORTS ---

logger = logging.getLogger(__name__)

# =====================================================
# 0️⃣ --- Global Configuration and Paths (UNCHANGED)
# =====================================================
BASE_DIR = os.path.normpath("C:/skinpredictor") 
SYMPTOM_MODEL_PATH = os.path.join(BASE_DIR, "disease_probability_model.pkl") 
YOLO_ONNX_PATH = os.path.join(BASE_DIR, "best.onnx") 
QUESTIONS_JSON_PATH = os.path.join(BASE_DIR, "questions.json") 
TARGET_SIZE = (640, 640) 
CONF_THRESHOLD = 0.5
SCORE_BOOST = 1.0 
YOLO_CLASSES = [
    'acne', 'rosacea', 'dark circles', 'pigmentation', 'wrinkles',
    'black-heads', 'sun spots', 'eye bags', 'freckles', 'skin cancer',
    'psoriasis', 'eczema', 'shingles', 'warts', 'hives', 'chicken pox'
]
ID_TO_CLASS = {i: name for i, name in enumerate(YOLO_CLASSES)}
CLASS_MAP = {name: i for i, name in ID_TO_CLASS.items()}
CONDITION_ALIASES = {k.lower(): k.lower() for k in CLASS_MAP.keys()}
MISCLASS_GROUPS = {
    "dark circles": ["eye bags"], "eye bags": ["dark circles"],
    "psoriasis": ["eczema"], "eczema": ["psoriasis"],
    "sun spots": ["pigmentation"], "pigmentation": ["sun spots"],
    "acne": ["chicken pox"], "rosacea": ["acne"],
    "chicken pox": ["acne"], "warts": ["skin cancer"],
    "skin cancer": ["warts"], "shingles": ["skin cancer"]
}
FALLBACK_SYMPTOMS = [
    # ... (Your existing list of fallback symptoms)
    "pustule", "papule", "tiny_bumps", "pain", "dark_bumps", "open_pores", "on_nose",
    "dark_crescents", "hollows_under_eyes", "sunken_eye_appearance", "mild_swelling_under_eyes",
    "puffiness_under_eyes", "tired_appearance", "small_flat_circular_spots", "appear_in_clusters",
    "darken_with_sun_exposure", "dark_patches_or_spots", "uneven_skin_tone", "worsen_with_sun_exposure",
    "new_or_changing_growths", "uneven_colour", "increases_in_size_or_thickness", "itching",
    "fine_lines_on_skin", "deeper_folds_or_creases", "loose_or_sagging_skin",
    "persistent_facial_redness", "visible_blood_vessels", "burning_or_stinging",
    "flat_oval_dark_spots", "even_in_texture", "well_defined_edges",
    "mild_burning_or_stinging", "dry_itchy_skin", "red_or_inflamed_patches", "severe_itching",
    "thick_raised_patches_of_skin", "mild_itching_or_burning", "pain_or_soreness", "well_defined_borders",
    "itchy_skin_rash_with_red_spots", "fluid_filled_blisters", "mild_fever", "fatigue_and_discomfort",
    "severe_itching_or_sensitivity_to_touch", "localized_pain", "red_rash_one_side",
    "raised_itchy_welts", "red_or_skin_coloured_welts", "intense_itching", "temporary_appearance",
    "small_rough_raised_bumps", "slow_growing", "rough_texture"
]
symptom_model = None
feature_names = None
label_encoder = None
detector = None
questions_data = None 

# =====================================================
# 1️⃣ --- Logistic Regression Prediction Helper (UNCHANGED)
# =====================================================
def predict_symptom_diseases(symptom_list, top_k=5, min_prob=0.30):
    # ... (Your existing LR prediction logic)
    if symptom_model is None or label_encoder is None or feature_names is None:
        logger.warning("Symptom model or resources not fully loaded. Skipping LR prediction.")
        return []
    full_vector = {feat: 0 for feat in feature_names}
    for s in (symptom_list or []):
        if s in full_vector:
            full_vector[s] = 1
    df = pd.DataFrame([full_vector])
    df = df.reindex(columns=feature_names, fill_value=0)
    try:
        probs = symptom_model.predict_proba(df)[0]
        class_names = list(label_encoder.classes_)
        idx_sorted = np.argsort(probs)[::-1]
        results = [(class_names[i], float(probs[i])) for i in idx_sorted]
        filtered = [(cls, p) for cls, p in results if p >= min_prob]
        return filtered[:top_k] if filtered else results[:top_k]
    except Exception as e:
        logger.error(f"Error during LR prediction: {e}")
        traceback.print_exc()
        return []

# =====================================================
# 2️⃣ --- Resource Loading (UNCHANGED)
# =====================================================
def load_all_resources():
    # ... (Your existing resource loading logic)
    global symptom_model, label_encoder, feature_names, detector, questions_data
    print(f"DEBUG: Starting resource load. Base Directory: {BASE_DIR}", file=sys.stderr)
    # 1. Load Logistic Regression Model
    try:
        symptom_model = joblib.load(SYMPTOM_MODEL_PATH)
        print(f"DEBUG: ✅ Loaded symptom model from {SYMPTOM_MODEL_PATH}", file=sys.stderr)
        if hasattr(symptom_model, "feature_names_in_"):
            feature_names = list(getattr(symptom_model, "feature_names_in_"))
        else:
            feature_names = FALLBACK_SYMPTOMS
            logger.warning("Feature names not found in LR model. Using fallback list.")
        if hasattr(symptom_model, "classes_"):
            class MockLabelEncoder:
                def __init__(self, classes):
                    self.classes_ = [str(c).lower() for c in classes] 
            label_encoder = MockLabelEncoder(symptom_model.classes_)
        else:
            logger.error("LR Model classes_ attribute not found.")
            label_encoder = None
    except FileNotFoundError:
        print(f"DEBUG: ❌ LR Model file NOT FOUND at: {SYMPTOM_MODEL_PATH}", file=sys.stderr)
        symptom_model, feature_names, label_encoder = None, None, None
    except Exception as e:
        print(f"DEBUG: ❌ Error loading LR model: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        symptom_model, feature_names, label_encoder = None, None, None
    # 3. Load YOLO Model
    try:
        detector = YOLO(YOLO_ONNX_PATH)
        print(f"DEBUG: ✅ Loaded YOLO model from {YOLO_ONNX_PATH}", file=sys.stderr)
    except FileNotFoundError:
        print(f"DEBUG: ❌ YOLO Model file NOT FOUND at: {YOLO_ONNX_PATH}", file=sys.stderr)
        detector = None
    except Exception as e:
        print(f"DEBUG: ❌ Error loading YOLO model: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        detector = None
    # 4. Load Questions JSON 
    try:
        with open(QUESTIONS_JSON_PATH, "r", encoding="utf-8") as f:
            questions_data = {k.lower(): v for k, v in json.load(f).items()}
        print(f"DEBUG: ✅ Loaded questions from {QUESTIONS_JSON_PATH}. Total conditions loaded: {len(questions_data)}", file=sys.stderr)
        if not questions_data:
            print("DEBUG: 🚨 questions_data is EMPTY after loading. Ensure JSON content is valid.", file=sys.stderr)
    except FileNotFoundError:
        print(f"DEBUG: ❌ CRITICAL ERROR: Questions JSON file NOT FOUND at: {QUESTIONS_JSON_PATH}", file=sys.stderr)
        questions_data = {}
    except Exception as e:
        print(f"DEBUG: ❌ CRITICAL ERROR: Error loading/parsing questions.json: {e}.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        questions_data = {}

load_all_resources()

# =====================================================
# 3️⃣ --- Follow-up Question Helpers (UNCHANGED)
# =====================================================
def _expand_by_misclass(detected_conditions):
    # ... (Your existing misclassification expansion logic)
    related = set(detected_conditions)
    for cond in detected_conditions:
        if cond in MISCLASS_GROUPS:
            related.update(MISCLASS_GROUPS[cond])
    return sorted(list(related))

def get_follow_up_questions(yolo_labels):
    """
    Returns a flat list of all follow-up questions for detected labels.
    Each question dict contains: condition, symptom_key, question
    """
    if not questions_data:
        logger.warning("Cannot generate questions: questions_data is empty.")
        return []

    related_conditions = _expand_by_misclass(yolo_labels)
    questions_to_ask = []
    seen_keys = set()

    def normalize_label(label):
        return label.lower().replace(" ", "_").strip()

    for cond in related_conditions:
        cond_key = normalize_label(cond)

        if cond_key not in questions_data:
            logger.warning(f"No questions found for condition: '{cond}' (normalized: '{cond_key}')")
            # fallback question if missing
            questions_to_ask.append({
                "condition": cond.capitalize(),
                "symptom_key": cond_key,
                "question": f"Do you have {cond}?"
            })
            continue

        cond_entry = questions_data[cond_key]
        cond_questions = cond_entry.get("questions", {})
        auto_confirm_text = cond_entry.get("auto_confirm", f"{cond.capitalize()} detected if any answer is Yes")

        if not cond_questions:
            # if questions dict empty, use auto_confirm text
            questions_to_ask.append({
                "condition": cond.capitalize(),
                "symptom_key": cond_key,
                "question": auto_confirm_text
            })
            continue

        # Flatten all questions
        for symptom, q_text in cond_questions.items():
            uid = f"{cond_key}_{symptom}"
            if uid in seen_keys:
                continue
            seen_keys.add(uid)
            questions_to_ask.append({
                "condition": cond.capitalize(),
                "symptom_key": symptom.strip(),
                "question": q_text.strip() if q_text else auto_confirm_text
            })

    logger.info(f"Generated {len(questions_to_ask)} follow-up questions for: {related_conditions}")
    return questions_to_ask

# =====================================================
# 4️⃣ --- Progress Tracking Helpers (NEW)
# =====================================================

def get_baseline_progress(user):
    """Retrieves the most recent record for comparison."""
    try:
        # Changed 'timestamp' to 'created_at' to match your model
        return SkinProgress.objects.filter(user=user).order_by('-created_at').first()
    except Exception as e:
        logger.error(f"Error retrieving baseline progress: {e}")
        return None

def analyze_progress(current_confidence_scores, baseline_progress):
    """Compares current detection against baseline."""
    # Ensure baseline has the necessary field
    if not baseline_progress or not hasattr(baseline_progress, 'confidence_scores_json') or not baseline_progress.confidence_scores_json:
        return {"status": "New analysis complete, no baseline data for comparison."}
    
    try:
        baseline_scores = json.loads(baseline_progress.confidence_scores_json)
    except Exception:
        return {"status": "New analysis complete, baseline data is corrupted."}
        
    improvements = {}
    regressions = {}
    
    for issue, current_score in current_confidence_scores.items():
        if issue in baseline_scores:
            baseline_score = baseline_scores[issue]
            score_change = current_score - baseline_score
            if score_change < -5.0:
                improvements[issue] = f"Reduced from {baseline_score:.1f}% to {current_score:.1f}%"
            elif score_change > 5.0:
                regressions[issue] = f"Increased from {baseline_score:.1f}% to {current_score:.1f}%"

    return {
        "status": "Comparison complete.",
        "improvements": improvements,
        "regressions": regressions
    }
def save_prediction_result(user, detected_issues, confidence_scores, image=None):
    """Save AI prediction results as a SkinProgress record."""
    avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0

    return SkinProgress.objects.create(
        user=user,
        image=image,
        detection_result=", ".join(detected_issues),
        ai_confidence=avg_confidence,
        confidence_scores_json=confidence_scores,
        improvement_score=0,           # keep default or calculate if needed
        age=getattr(user, 'age', 0),  # pull from user profile
        gender=getattr(user, 'gender', 'Not specified')
    )


from django.core.files.base import ContentFile
from users.models import MyAIReport

@login_required
@csrf_exempt
@require_POST
# -----------------------------
# PREDICT VIEW
# -----------------------------
@login_required
def predict_view(request):
    """Handle image prediction with YOLO, save progress and user AI report snapshot."""
    import io, base64, json
    from django.core.files.base import ContentFile
    from PIL import Image, ImageDraw, ImageFont
    from django.utils import timezone
    from users.models import MyAIReport
    
    # from users.utils import save_prediction_result, get_baseline_progress, analyze_progress, predict_symptom_diseases, get_follow_up_questions

    if detector is None:
        return JsonResponse({'error': 'YOLO model not loaded.'}, status=503)
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)

    image_file = request.FILES['file']

    # Load image
    try:
        img_original = Image.open(image_file).convert("RGB")
    except Exception:
        return JsonResponse({'error': 'Invalid image file.'}, status=400)

    # YOLO prediction
    try:
        results = detector.predict(
            source=img_original,
            conf=CONF_THRESHOLD,
            iou=0.7,
            imgsz=TARGET_SIZE[0],
            verbose=False
        )
    except Exception:
        return JsonResponse({'error': 'YOLO prediction failed'}, status=500)

    # Process YOLO results
    detected_issues = set()
    confidence_scores = {}
    all_boxes, all_labels, all_scores = [], [], []

    if results and len(results) > 0:
        r = results[0]
        try:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            boxes, confs, cls_ids = [], [], []

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            label = ID_TO_CLASS.get(int(cls_id), 'unknown').lower()
            score = min(1.0, float(conf) * SCORE_BOOST)
            if score < CONF_THRESHOLD:
                continue
            detected_issues.add(label)
            confidence_scores[label] = max(confidence_scores.get(label, 0), round(score * 100, 2))
            all_boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            all_labels.append(label)
            all_scores.append(round(score * 100, 2))

    # Front-end data
    data = json.loads(request.POST.get('data', '{}'))
    answers_submitted = data.get('answers_submitted', False)
    user_symptoms = data.get('symptoms', [])
    confirmed_conditions = [c.lower() for c in data.get('confirmed_conditions', [])]
    follow_up_answers = data.get('follow_up_answers', [])

    # Expand detected issues for misclassified groups
    all_issues = set(detected_issues)
    for issue in detected_issues:
        if issue in MISCLASS_GROUPS:
            all_issues.update(MISCLASS_GROUPS[issue])

    # Handle follow-up questions
    follow_up_questions = []
    if not answers_submitted and all_issues:
        follow_up_questions = get_follow_up_questions(list(all_issues))
        if follow_up_questions:
            img_draw = img_original.copy()
            draw = ImageDraw.Draw(img_draw)
            try:
                font = ImageFont.truetype("arial.ttf", 50)
            except:
                font = ImageFont.load_default()
            for box, label, score in zip(all_boxes, all_labels, all_scores):
                for t in range(10):
                    draw.rectangle([box[0]-t, box[1]-t, box[2]+t, box[3]+t], outline="red")
                draw.text((box[0], max(0, box[1]-25)), f"{label.capitalize()} ({score}%)", fill="white", font=font)
            img_io = io.BytesIO()
            img_draw.save(img_io, format='JPEG')
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            return JsonResponse({
                'status': 'success',
                'annotated_image': img_base64,
                'detected_issues': list(all_issues),
                'follow_up_questions': follow_up_questions,
                'confidence_scores': confidence_scores,
                'progress_summary': {"status": "Please answer follow-up questions to refine diagnosis."},
                'remedies_data': {}
            })

    # Add follow-up answers & confirmed conditions
    for ans in follow_up_answers:
        condition_name = ans.get('condition', '').lower()
        if ans.get('answer', '').lower() == 'yes' and condition_name:
            all_issues.add(condition_name)
            confidence_scores[condition_name] = 90.0
    for cond in confirmed_conditions:
        all_issues.add(cond)
        confidence_scores[cond] = max(confidence_scores.get(cond, 0), 90.0)

    # Refine using symptoms
    if user_symptoms:
        symptom_preds = predict_symptom_diseases(user_symptoms, top_k=5, min_prob=0.30)
        for disease, prob in symptom_preds:
            if prob * 100 >= 30:
                all_issues.add(disease.lower())
                confidence_scores[disease.lower()] = round(prob * 100, 2)

    # Annotate image
    img_draw = img_original.copy()
    draw = ImageDraw.Draw(img_draw)
    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except:
        font = ImageFont.load_default()
    for box, label, score in zip(all_boxes, all_labels, all_scores):
        for t in range(10):
            draw.rectangle([box[0]-t, box[1]-t, box[2]+t, box[3]+t], outline="red")
        draw.text((box[0], max(0, box[1]-25)), f"{label.capitalize()} ({score}%)", fill="white", font=font)

    img_io = io.BytesIO()
    img_draw.save(img_io, format='JPEG')
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    annotated_file = ContentFile(img_io.getvalue(), name=f"annotated_{timezone.now().strftime('%Y%m%d%H%M%S')}.jpg")

    # Remedies
    remedies_data = {}
    for issue in all_issues:
        normalized_issue = CONDITION_ALIASES.get(issue.lower(), issue.lower())
        condition = SkinCondition.objects.filter(name__iexact=normalized_issue).first()
        if not condition:
            continue

        home_remedies = [
            {
                'title': r.title,
                'directions': r.directions or "",
                'amount': r.amount or "",
                'image_url': r.image.url if r.image else None
            } for r in condition.remedy_set.all()
        ]
        medical_remedies = [
            {
                'title': t.title,
                'directions': t.directions or "",
                'amount': t.amount or "",
                'image_url': t.image.url if t.image else None,
                'scientific_evidence': t.scientific_evidence
            } for t in condition.treatment_set.filter(category='medical')
        ]

        remedies_data[normalized_issue] = {
            'home_remedies': home_remedies,
            'medical_remedies': medical_remedies,
            'causes': [c.strip() for c in getattr(condition, 'causes', '').split('\n') if c.strip()],
            'symptoms': [s.strip() for s in getattr(condition, 'symptoms', '').split('\n') if s.strip()]
        }

    # Save SkinProgress & MyAIReport for user (doctor=None)
    progress_summary = None
    if request.user.is_authenticated:
        skin_progress = save_prediction_result(request.user, list(all_issues), confidence_scores, image=image_file)

        MyAIReport.objects.create(
            user=request.user,
            doctor=None,  # doctor linked later
            detected_issues=list(all_issues),
            confidence_scores=int(sum(confidence_scores.values())/len(confidence_scores) if confidence_scores else 0),
            confidence_scores_json=confidence_scores,
            annotated_image=annotated_file,
            skin_progress=skin_progress,
            prediction=", ".join(list(all_issues)),
            age=getattr(request.user, 'age', None),       # <-- add this
            gender=getattr(request.user, 'gender', None)  # <-- and this
        )


        # Analyze baseline progress
        baseline = get_baseline_progress(request.user)
        if baseline:
            progress_summary = analyze_progress(confidence_scores, baseline)

    return JsonResponse({
        'status': 'success',
        'annotated_image': img_base64,
        'detected_issues': list(all_issues),
        'remedies_data': remedies_data,
        'confidence_scores': confidence_scores,
        'follow_up_questions': [],  # already answered
        'progress_summary': progress_summary or {"status": "Analysis complete."}
    })





from django.http import HttpResponse
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListItem, ListFlowable
from .models import PersonalizedPlan

# Map your URL slugs to your Database condition_names
CONDITION_NAME_MAP = {
    "acne": "Acne",
    "rosacea": "Rosacea",
    "dark-circles": "Dark Circles",
    "pigmentation": "Pigmentation (Hyperpigmentation)",
    "wrinkles": "Wrinkles & Fine Lines",
    "blackheads": "Blackheads (Open Comedones)",
    "sun-spots": "Sun Spots (Solar Lentigines)",
    "eye-bags": "Eye Bags & Puffiness",
    "freckles": "Freckles (Ephelides)",
    "skin-cancer": "Skin Cancer Prevention & Post-Diagnosis Care",
    "psoriasis": "Psoriasis",
    "eczema": "Eczema (Atopic Dermatitis)",
    "shingles": "Shingles (Herpes Zoster)",
    "warts": "Warts (Common & Plantar)",
    "hives": "Hives (Urticaria)",
    "chicken-pox": "Chicken Pox (Varicella)"
} 

# Define a Professional Green Color Palette
DARK_GREEN = colors.HexColor("#376740")  # For Main Title
MEDIUM_GREEN = colors.HexColor("#209203") # For Condition Name
LIGHT_GREEN = colors.HexColor("#50aa1f")  # For Accent lines and Bullets

def download_lifestyle(request):
    conditions_param = request.GET.get('conditions', '')
    slugs = conditions_param.split(',') if conditions_param else []
    
    # Map slugs to DB names
    db_names = [CONDITION_NAME_MAP.get(s) for s in slugs if CONDITION_NAME_MAP.get(s)]
    
    # Fetch data from DB
    plans = PersonalizedPlan.objects.filter(condition_name__in=db_names)

    buffer = BytesIO()
    # SimpleDocTemplate handles page margins and flow automatically
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    story = []

    # --- CUSTOM STYLES ---
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=DARK_GREEN,
        spaceAfter=12,
        fontName="Helvetica-Bold"
    )
    
    condition_style = ParagraphStyle(
        'CondHeader',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=MEDIUM_GREEN,
        spaceBefore=15,
        spaceAfter=10,
        fontName="Helvetica-Bold"
    )

    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=DARK_GREEN,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14, # Line spacing
        textColor=colors.black
    )

    # --- BUILDING THE PDF CONTENT ---
    # Main Header
    story.append(Paragraph("Personalized Lifestyle Plan", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=LIGHT_GREEN, spaceAfter=20))

    if not plans.exists():
        story.append(Paragraph("No lifestyle plans found for selected conditions.", body_style))
    else:
        for plan in plans:
            # Condition Title
            story.append(Paragraph(f"Condition: {plan.condition_name}", condition_style))

            # Define the sections based on your model fields
            sections = [
                ("Dietary Advice", plan.diet),
                ("Skincare Routine", plan.skincare),
                ("Exercise Tips", plan.exercise),
                ("Sleep & Habits", plan.sleep)
            ]

            for title, content in sections:
                if content:
                    # Section Heading
                    story.append(Paragraph(title, section_style))
                    
                    # Process lines into a bulleted list
                    lines = str(content).split('\n')
                    bullet_items = []
                    for line in lines:
                        if line.strip():
                            # This creates the "-" bullet look with green color
                            bullet_items.append(
                                ListItem(
                                    Paragraph(line.strip(), body_style),
                                    bulletColor=LIGHT_GREEN,
                                    value="-"
                                )
                            )
                    
                    if bullet_items:
                        story.append(ListFlowable(bullet_items, bulletType='bullet', leftIndent=20))
                    
                    story.append(Spacer(1, 10))

            # Add space or Page Break between different conditions
            story.append(Spacer(1, 30))

    # Build PDF
    doc.build(story)
    
    pdf = buffer.getvalue()
    buffer.close()
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="lifestyle_plan.pdf"'
    response.write(pdf)
    return response


def lifestyle_plan_view(request, condition_name):
    """
    Returns the personalized lifestyle plan for a detected condition.
    """
    # Ensure proper capitalization / formatting if needed
    condition_name = condition_name.lower()  # 'acne', 'rosacea', etc.

    # Fetch the plan from DB
    plan = get_object_or_404(PersonalizedPlan, condition_name__iexact=condition_name)

    context = {
        'plan': plan
    }
    return render(request, 'lifestyle_pdf.html', context)



def capture(request):
    # This view function captures a single image from a webcam.
    cap = cv2.VideoCapture(0) # Initializes the video capture from the default camera (index 0).

    if not cap.isOpened():
        return HttpResponse("Webcam not accessible", status=500) # Returns an error if the camera can't be opened.

    ret, frame = cap.read() # Reads a single frame from the camera.
    cap.release() # Releases the camera resource.

    if not ret:
        return HttpResponse("Failed to capture image", status=500) # Returns an error if the capture failed.

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converts the frame from BGR (OpenCV default) to RGB.
    img_pil = Image.fromarray(frame_rgb) # Converts the NumPy array to a PIL Image.

    img_io = io.BytesIO() # Creates an in-memory byte stream.
    img_pil.save(img_io, format='JPEG') # Saves the image to the stream.
    img_io.seek(0) # Resets the stream pointer.

    return HttpResponse(img_io, content_type="image/jpeg") # Returns the image as an HTTP response.

def get_remedies(request):
    """
    Fetches and returns remedies (home + medical) and details for a specific skin condition.
    """
    issue = request.GET.get('issue', '').lower()  # Get the 'issue' parameter from URL

    try:
        condition = SkinCondition.objects.get(name__iexact=issue)

        # Causes and symptoms
        causes = [c.strip() for c in (condition.causes.split('\n') if condition.causes else []) if c.strip()]
        symptoms = [s.strip() for s in (condition.symptoms.split('\n') if condition.symptoms else []) if s.strip()]

        # Home remedies (from SkinCondition)
        home_remedies = [
            {
                "title": r.title,
                "directions": getattr(r, 'formatted_directions', lambda: r.directions)(),
                "amount": r.amount or "",
                "image_url": r.image.url if r.image else None
            }
            for r in condition.remedy_set.all()
        ]

        # Medical treatments (from Treatment model)
        medical_remedies = [
            {
                "title": t.title,
                "directions": t.directions or "",
                "amount": t.amount or "",
                "image_url": t.image.url if t.image else None,
                "scientific_evidence": t.scientific_evidence
            }
            for t in condition.treatment_set.filter(category='medical')
        ]

        # Debug log
        print(f"Sending remedies for '{issue}': home={len(home_remedies)}, medical={len(medical_remedies)}")

        return JsonResponse({
            "causes": causes,
            "symptoms": symptoms,
            "home_remedies": home_remedies,
            "medical_remedies": medical_remedies
        })

    except SkinCondition.DoesNotExist:
        return JsonResponse({'error': 'Condition not found'}, status=404)

# Your other view functions (e.g., user registration, login, etc.)
# ...

@login_required
def predict_page_view(request):
    # This view renders the main prediction page template.
    return render(request, 'predict.html')


from utils.aliases import CONDITION_ALIASES; # Re-importing a module; this line might be redundant if it's already at the top.

def article_general_view(request):
    # This view renders a generic article page template.
    return render(request, 'article.html', {}) # Renders the template with an empty context.

def skin_conditions_list_view(request):
    # This view fetches a list of all skin conditions from the database and renders a list page.
    all_skin_conditions = SkinCondition_page.objects.all() # Fetches all objects of the 'SkinCondition_page' model.

    context = {
        'all_skin_conditions': all_skin_conditions,
    }
    return render(request, 'Skin Conditions.html', context) # Renders the template with the list of conditions.


def skin_condition_detail(request, condition_slug):
    # This view fetches and displays a single skin condition's details based on its slug.
    skin_condition = get_object_or_404(SkinCondition_page, slug=condition_slug) # Fetches the object or returns a 404 error if not found.

    context = {
        'skin_condition': skin_condition,
    }
    return render(request, 'Skin Conditions.html', context) # Renders the detail page template.

from .models import Article # Imports the 'Article' model.

def article_detail(request, slug):
    # This view fetches a specific article by its slug and displays it.
    article = get_object_or_404(Article, slug=slug) # Fetches the article or returns a 404.

    # --- ADD THESE DEBUGGING LINES ---
    print(f"DEBUG: Type of 'article': {type(article)}")
    print(f"DEBUG: Content of 'article': {article}")
    if hasattr(article, 'pk'):
        print(f"DEBUG: PK of 'article': {article.pk}")
    else:
        print("DEBUG: 'article' object has no 'pk' attribute.")
    # --- END DEBUGGING LINES ---
    # These are temporary debugging lines to check the fetched object's details.

    related_articles = Article.objects.filter(
        category=article.category
    ).exclude(pk=article.pk)[:3] # Fetches up to 3 other articles from the same category, excluding the current one.

    return render(request, 'article_detail.html', {
        'article': article,
        'related_articles': related_articles
    }) # Renders the article detail page with the main article and related articles.




























