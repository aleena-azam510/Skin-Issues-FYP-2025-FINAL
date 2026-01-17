# views.py updates

from datetime import timezone
import json
from django.http import JsonResponse
from django.shortcuts import render
from predictor.views import predict_view
from users.models import SkinProgress
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from users.models import SkinProgress
import logging

# Standard way to get a logger instance for the current module
logger = logging.getLogger(__name__)

def get_skin_stats(user):
    """Get skin tracking statistics for a user"""
    uploads = SkinProgress.objects.filter(user=user).order_by('-created_at')
    
    # Calculate improvements for each upload
    uploads_with_improvement = []
    for i, upload in enumerate(uploads):
        improvement_data = calculate_improvement_score(upload, uploads[i+1] if i+1 < len(uploads) else None)
        upload.improvement_score = improvement_data['score']
        upload.improvement_trend = improvement_data['trend']
        uploads_with_improvement.append(upload)
    
    # Get latest result for display
    latest_result = None
    if uploads:
        latest = uploads.first()
        latest_result = {
            'confidence': latest.ai_confidence,
            'conditions': latest.detection_result.split(", ") if latest.detection_result else [],
            'image_url': latest.image.url,
            'date': latest.created_at.strftime("%b %d, %Y"),
            'improvement': getattr(latest, 'improvement_score', 0),
        }
    
    return {
        'uploads': uploads_with_improvement,
        'latest_result': latest_result,
        'total_analyses': uploads.count(),
    }

def calculate_improvement_score(current, previous=None):
    """Calculate improvement score between current and previous analysis"""
    if not previous:
        return {'score': 0, 'trend': 'First analysis'}
    
    # Calculate improvement based on confidence changes
    current_confidence = current.ai_confidence or 0
    previous_confidence = previous.ai_confidence or 0
    
    # Simple improvement calculation (can be enhanced)
    if current.detection_result and previous.detection_result:
        current_issues = set(current.detection_result.split(", "))
        previous_issues = set(previous.detection_result.split(", "))
        
        # Issues resolved
        resolved = previous_issues - current_issues
        # New issues
        new_issues = current_issues - previous_issues
        
        improvement_score = (len(resolved) * 20) - (len(new_issues) * 10)
        improvement_score += (current_confidence - previous_confidence) / 10
        
        # Cap the score
        improvement_score = max(-100, min(100, improvement_score))
        
        trend = []
        if len(resolved) > 0:
            trend.append(f"Resolved {len(resolved)} issues")
        if len(new_issues) > 0:
            trend.append(f"New {len(new_issues)} issues")
        if current_confidence > previous_confidence:
            trend.append("Confidence improved")
            
        return {
            'score': round(improvement_score, 1),
            'trend': " | ".join(trend) if trend else "No significant change"
        }
    
    return {'score': 0, 'trend': 'Insufficient data'}

@login_required
def skin_tracker(request):
    if request.method == "POST":
        # Handle file upload
        image = request.FILES.get("image")
        if not image:
            return JsonResponse({"status": "error", "message": "No image uploaded"}, status=400)
        
        try:
            # Call the prediction view
            request.FILES["file"] = image
            response = predict_view(request)
            
            if response.status_code != 200:
                return JsonResponse({"status": "error", "message": "AI analysis failed"}, status=500)
            
            result = json.loads(response.content)
            
            # Get or create the latest record
            latest = SkinProgress.objects.filter(user=request.user).order_by('-created_at').first()
            
            # Calculate improvement if we have previous data
            improvement = 0
            if latest and hasattr(latest, 'ai_confidence'):
                previous_confidence = latest.ai_confidence or 0
                current_confidence = result.get('confidence_scores', {})
                avg_confidence = sum(current_confidence.values()) / len(current_confidence) if current_confidence else 0
                improvement = round((avg_confidence - previous_confidence) * 0.1, 1)
            
            return JsonResponse({
                "status": "success",
                "data": {
                    "confidence": round(avg_confidence, 1),
                    "conditions": result.get('detected_issues', []),
                    "image_url": f"/media/skin_progress/{request.user.id}/{image.name}",  # Adjust path as needed
                    "date": timezone.now().strftime("%b %d, %Y"),
                    "improvement": improvement,
                }
            })
            
        except Exception as e:
            logger.exception(f"Skin tracker error: {str(e)}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    # GET request - render with stats
    context = get_skin_stats(request.user)
    return render(request, "dashboard/skin_tracker.html", context)

@login_required
@require_POST
def delete_skin_image(request, id):
    """Delete skin image with proper JSON response"""
    try:
        obj = SkinProgress.objects.get(id=id, user=request.user)
        
        # Delete the image file
        if obj.image:
            obj.image.delete(save=False)
        
        # Delete the database record
        obj.delete()
        
        return JsonResponse({"success": True})
        
    except SkinProgress.DoesNotExist:
        return JsonResponse({"success": False, "error": "Record not found"}, status=404)
        
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        return JsonResponse({"success": False, "error": str(e)}, status=500)