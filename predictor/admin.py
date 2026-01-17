# from django.contrib import admin

# # Register your models here.
# from .models import SkinCondition

# admin.site.register(SkinCondition)


from django.contrib import admin
from .models import SkinCondition, Remedy,SkinCondition_page , PersonalizedPlan
from .models import Article



class RemedyInline(admin.TabularInline):
    model = Remedy
    extra = 3  # Display space for 3 remedies
    fields = ('title', 'image', 'image_preview', 'amount', 'directions')  # Updated fields
    readonly_fields = ('image_preview',)  # Make preview read-only
    
    def image_preview(self, obj):
        return obj.image_preview()
    
    image_preview.short_description = 'Preview'
    
class SkinConditionAdmin(admin.ModelAdmin):
    inlines = [RemedyInline]
    list_display = ('name',)
    search_fields = ('name',)

admin.site.register(SkinCondition, SkinConditionAdmin)
admin.site.register(SkinCondition_page)

from django import forms
# Correct import based on your forms.py:
from jsoneditor.forms import JSONEditor 
from .models import Article  # ✅ You need this import


class ArticleAdminForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = '__all__'
        widgets = {
            'content': JSONEditor(
                init_options={
                    'mode': 'code', # Use 'code' or 'tree'
                    'modes': ['code', 'tree', 'form', 'text', 'view'], # Available modes
                },
                # You can add ace_options here if you want to customize the ACE editor
                # ace_options={
                #     'showLineNumbers': True,
                #     'tabSize': 2,
                # }
            ),
        }

@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    form = ArticleAdminForm
    list_display = ('title', 'author', 'category', 'published_date', 'is_featured', 'reading_time')
    list_filter = ('category', 'is_featured', 'published_date')
    search_fields = ('title', 'excerpt')
    prepopulated_fields = {'slug': ('title',)}
    date_hierarchy = 'published_date'
    
    
    
from .models import Treatment

@admin.register(Treatment)
class TreatmentAdmin(admin.ModelAdmin):
    list_display = ['title', 'skin_condition', 'category', 'scientific_evidence', 'amount', 'created_at']
    list_filter = ['category', 'scientific_evidence']
    search_fields = ['title', 'skin_condition__name']

@admin.register(PersonalizedPlan)
class PersonalizedPlanAdmin(admin.ModelAdmin):
    list_display = ('condition_name', 'created_at')