from django.contrib import admin
from django.urls import path
from . import views
# In urls.py
from django.conf import settings
from django.conf.urls.static import static


app_name = 'predictor' # <--- Make sure this line exists and is correct

urlpatterns = [
        path('admin/', admin.site.urls),
    # Other pages remain accessible
        #path('index/', views.index_view, name='index_page'), # Still accessible if needed
        #path('skin-conditions-list/', views.skin_conditions_list_view, name='skin_conditions_list'),
        path('article/', views.article_general_view, name='article_page'), # <--- ADD THIS LINE
        path('article/<slug:slug>/', views.article_detail, name='article_detail'),
        path('skin-conditions-list/', views.skin_conditions_list_view, name='skin_conditions_list'),

    # Path for the individual skin condition detail page, using the slug
    # This URL will typically be something like /skin-conditions/acne/
        path('skin-conditions/<slug:condition_slug>/', views.skin_condition_detail, name='skin_condition_detail'),

        # path('', views.home, name='home'),  # For showing the predict page
         # Path to render the predict HTML page
        path('predict/', views.predict_page_view, name='predict_page'), 
        
        path('api/predict/', views.predict_view, name='predict_api'),  # Predict API
        path('capture/', views.capture, name='capture'),  
# skinpredictor/urls.py
    path('download-personalized-plan/', views.download_lifestyle, name='download_personalized_plan'),        # path('skin-conditions/<slug:condition_slug>/', views.skin_condition_detail_view, name='skin_condition_detail'),
    # path("send-report-to-doctor/", views.send_report_to_doctor, name="send_report_to_doctor"),
    


        
        #  path('api/remedies/', views.get_remedies, name='get_remedies'),
]
if settings.DEBUG:
#     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)