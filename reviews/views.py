# reviews/views.py

from django.http import JsonResponse
from django.shortcuts import render # Keep render if you use it for GET requests, but not for POST in this AJAX setup
from django.contrib import messages # Good for non-AJAX paths, but not used directly by our AJAX success/error logic


from .models import Review

def submit_review(request):
    if request.method == 'POST':
        reviewer_name = request.POST.get('reviewer_name')
        rating_str = request.POST.get('rating')
        review_text = request.POST.get('review_text')

        errors = {}

        # --- VALIDATION SECTION ---
        if not reviewer_name or reviewer_name.strip() == '':
            errors['reviewer_name'] = ["Your name is required."]
        if not rating_str:
            errors['rating'] = ["A rating is required."]
        if not review_text or review_text.strip() == '':
            errors['review_text'] = ["Your review text is required."]

        rating = None
        try:
            rating = int(rating_str)
            if not (1 <= rating <= 5):
                # Use .get() with a default list to append multiple errors for the same field
                errors['rating'] = errors.get('rating', []) + ["Rating must be between 1 and 5."]
        except (ValueError, TypeError):
            errors['rating'] = errors.get('rating', []) + ["Invalid rating provided. Must be a number."]


        if errors:
            return JsonResponse({'success': False, 'errors': errors})

        try:
            review = Review(
                reviewer_name=reviewer_name.strip(),
                rating=rating,
                review_text=review_text.strip()
            )
            if request.user.is_authenticated:
                review.user = request.user
            review.save()

            # --- ADD THIS LINE TO GET THE UPDATED COUNT ---
            updated_total_reviews_count = Review.objects.count()

            return JsonResponse({
                'success': True,
                'message': 'Thank you for your review! It has been submitted successfully.',
                'total_reviews_count': updated_total_reviews_count # Include the count
            })

        except Exception as e:
            return JsonResponse({'success': False, 'errors': {'server_error': [f"An unexpected error occurred: {e}"]}})
    else:
        return JsonResponse({'success': False, 'errors': {'method_error': ["GET method not allowed for this endpoint."]}}, status=405)
    
def get_reviews_list(request):
    """
    Returns the reviews list HTML for AJAX updates.
    """
    reviews = Review.objects.all().order_by('-created_at') # Order by newest first
    # You might want to paginate this in a real application
    return render(request, 'reviews/partials/reviews_list_partial.html', {'reviews': reviews})
    
from django.core.mail import send_mail # <--- Add this line   
from .forms import ContactForm # Import your new ContactForm
from django.conf import settings # <--- Add this line


# ... (your existing submit_review view) ...

def contact_form_submit(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']

            try:
                # Send email
                send_mail(
                    f'Contact Form Message from {name}: {subject}', # Email subject
                    f'Name: {name}\nEmail: {email}\n\nMessage:\n{message}', # Email body
                    settings.DEFAULT_FROM_EMAIL, # From email (configured in settings.py)
                    [settings.CONTACT_EMAIL], # To email (your email address)
                    fail_silently=False,
                )
                return JsonResponse({'success': True, 'message': 'Your message has been sent successfully. Thank you!'})
            except Exception as e:
                # Log the error for debugging
                print(f"Error sending email: {e}")
                return JsonResponse({'success': False, 'errors': {'server_error': ['There was an error sending your message. Please try again later.']}})
        else:
            # Form is not valid, return validation errors
            return JsonResponse({'success': False, 'errors': form.errors})
    else:
        return JsonResponse({'success': False, 'errors': {'method_error': ["GET method not allowed for this endpoint."]}}, status=405)