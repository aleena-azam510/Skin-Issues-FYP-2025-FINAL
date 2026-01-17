from django.shortcuts import render
from .forms import SkinTypeForm
from .utils import diagnose_skin_type

def process_skin_data(form_data):
    """
    Translates diverse form data into a structured format for the skin type diagnosis utility.
    """
    # Initialize the data dictionary with default values
    diagnoser_data = {
        'oiliness': 'normal',
        'dryness': 'no',
        'sensitivity': 'no',
        'acne': 'no',
        'aging': 'no',
        # Adding new fields to initialize the diagnoser_data dictionary
        'texture': 'smooth',
        'environment': 'temperate',
        'hormonal_influence': 'none'
    }

    # Map core skin characteristics
    skin_feel = form_data.get('skin_feel')
    if skin_feel == 'shiny':
        diagnoser_data['oiliness'] = 'oily'
    elif skin_feel == 'combination':
        diagnoser_data['oiliness'] = 'combination'
    elif skin_feel == 'tight':
        diagnoser_data['dryness'] = 'frequent'
    elif skin_feel == 'comfortable':
        diagnoser_data['oiliness'] = 'normal'
        diagnoser_data['dryness'] = 'rarely'

    # Map skin concerns. Use `any()` for concise logic.
    skin_concerns = form_data.get('skin_concerns', [])
    if 'acne' in skin_concerns or 'blackheads' in skin_concerns:
        diagnoser_data['acne'] = 'frequent'
    if 'aging' in skin_concerns:
        diagnoser_data['aging'] = 'noticeable'
    if 'dryness' in skin_concerns:
        diagnoser_data['dryness'] = 'frequent'
    if 'redness' in skin_concerns:
        diagnoser_data['sensitivity'] = 'somewhat'

    # Map sensitivity level, overriding previous 'redness' assumption if more specific data is available.
    sensitivity_level = form_data.get('sensitivity_level')
    if sensitivity_level == 'very_sensitive':
        diagnoser_data['sensitivity'] = 'very'
    elif sensitivity_level == 'somewhat_sensitive':
        diagnoser_data['sensitivity'] = 'somewhat'

    # --- Mapping for the new, diverse data points ---
    
    # Map skin texture
    skin_texture = form_data.get('skin_texture')
    if skin_texture in ['uneven_rough', 'bumpy_clogged', 'flaky_patchy']:
        diagnoser_data['texture'] = 'uneven'

    # Map environmental factors (climate)
    climate = form_data.get('climate')
    if climate in ['dry', 'cold']:
        # Enhance dryness diagnosis if living in a dry or cold climate
        if diagnoser_data['dryness'] != 'frequent':
            diagnoser_data['dryness'] = 'seasonal' # A new state for contextual dryness
    elif climate == 'humid':
        # Enhance oiliness diagnosis in a humid climate
        if diagnoser_data['oiliness'] != 'oily':
            diagnoser_data['oiliness'] = 'combination' # Or at least combination

    # Map hormonal factors
    hormonal_factors = form_data.get('hormonal_factors', [])
    if 'pre_cycle' in hormonal_factors or 'stress_related' in hormonal_factors:
        diagnoser_data['hormonal_influence'] = 'high'
        # Hormonal influence often leads to acne
        if diagnoser_data['acne'] != 'frequent':
            diagnoser_data['acne'] = 'occasional'

    # Map product reactions to sensitivity and acne
    product_reaction = form_data.get('product_reaction')
    if product_reaction in ['irritation', 'redness']:
        if diagnoser_data['sensitivity'] != 'very':
            diagnoser_data['sensitivity'] = 'somewhat'
    elif product_reaction == 'breakouts':
        if diagnoser_data['acne'] != 'frequent':
            diagnoser_data['acne'] = 'occasional'

    # Other fields like `pore_size`, `diet_factors`, `current_routine`, and `skincare_routine_length` are useful
    # but don't directly map to the simple `diagnoser_data` keys. They provide rich context for a more
    # advanced diagnosis utility or for a human expert to review.

    return diagnoser_data


def skin_identifier_view(request):
    """
    Handles the skin type identification form and displays the results.
    """
    if request.method == 'POST':
        form = SkinTypeForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            diagnoser_data = process_skin_data(form_data)
            
            result = diagnose_skin_type(diagnoser_data)
            
            # Use redirect after a successful POST to prevent duplicate submissions
            # The result data will need to be stored in the session or a database
            # to be accessed on the new page, or you can pass it directly.
            # For simplicity, returning a render is fine if you're not
            # worried about the POST/redirect/GET pattern.
            return render(request, 'skin_identifier/result.html', {'result': result})
        else:
            # If the form is invalid, re-render the form page with the errors
            # The form object will contain the errors, which the template can display
            return render(request, 'skin_identifier/form.html', {'form': form})
    else:
        # This handles the initial GET request to show the empty form
        form = SkinTypeForm()
        
    return render(request, 'skin_identifier/form.html', {'form': form})