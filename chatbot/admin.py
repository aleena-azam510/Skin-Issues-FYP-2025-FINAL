# chatbot/admin.py
from django.contrib import admin
from django import forms
# from jsoneditor.forms import JSONEditor  # Assuming this is correct based on your Article model
from jsoneditor.forms import JSONEditor # Keep this if the previous one worked for Article. Otherwise, try .widgets

from .models import Answer, Question

class AnswerAdminForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = '__all__'
        widgets = {
            'content': JSONEditor(
                init_options={
                    'mode': 'code',
                    'modes': ['code', 'tree', 'form', 'text', 'view'],
                },
                # ace_options for syntax highlighting, etc.
                # ace_options={'showLineNumbers': True, 'tabSize': 2}
            ),
        }

    # IMPORTANT: Ensure these are commented out or removed!
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     if self.instance.pk and self.instance.content is not None:
    #         self.initial['content'] = json.dumps(self.instance.content, indent=2)
    #     else:
    #         self.initial['content'] = json.dumps([])

    # def clean_content(self):
    #     # ... (remove this entire method)
    #     pass


@admin.register(Answer)
class AnswerAdmin(admin.ModelAdmin):
    form = AnswerAdminForm
    list_display = ('__str__',)

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('text', 'answer',)
    list_filter = ('answer',)
    search_fields = ('text',)