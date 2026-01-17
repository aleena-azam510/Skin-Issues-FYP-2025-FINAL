from django.urls import path
from . import views

urlpatterns = [
    # ----------------------------
    # Public Pages
    # ----------------------------
    path('', views.index_page, name='index_page'),
    path('notifications/', views.notifications_view, name='notifications'),
    path('notifications/mark-read/<int:notification_id>/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/mark-all-read/', views.mark_all_notifications_read, name='mark_all_notifications_read'),
    path('notifications/clear-all/', views.clear_all_notifications, name='clear_all_notifications'),  # <-- add this
    path('developerTeam/', views.developer_team_view, name='developer_team_page'),
    path('auth/', views.auth_view, name='auth_page'),
    path('verify-email/<str:uidb64>/<str:token>/', views.verify_email, name='verify_email'),
    path('modelPage/', views.model_page_view, name='model_page'),
    path('article/', views.article_view, name='article_page'),

    # ----------------------------
    # User Dashboard
    # ----------------------------
    path('dashboard/', views.user_dashboard_home, name='user_dashboard_home'),
    path('dashboard/tracker/', views.skin_tracker, name='skin_tracker'),
    path('dashboard/tracker/delete/<int:id>/', views.delete_skin_image, name='delete_skin_image'),
    path('dashboard/tracker/delete_all/', views.delete_all_skin_history, name='delete_all_skin_history'),
    path('dashboard/uv/', views.uv_protection, name='uv_protection'),
    path('dashboard/lifestyle/', views.lifestyle, name='lifestyle'),
    path('dashboard/sleep/', views.sleep_stress, name='sleep_stress'),
    path('dashboard/habits/', views.habits, name='habits'),
    path('dashboard/journal/', views.journal, name='journal'),
    # users/urls.py
    path('dashboard/my-ai-reports/', views.my_ai_reports, name='my_ai_reports'),
    path('dashboard/reviews/', views.user_reviews, name='reviews'),

    path('dashboard/redirect/', views.dashboard_redirect, name='dashboard_redirect'),
    path('dashboard/my-ai-reports/delete/<int:report_id>/', views.delete_ai_report, name='delete_ai_report'),
  
   # -------- --------------------
    # Doctor Pages
    # ----------------------------
    path('doctor/dashboard/', views.doctor_home, name='doctor_home'),

    # Doctor creates / edits OWN profile
    path(
        'doctor/profile/',
        views.doctor_profile_manage,
        name='doctor_profile'
    ),

    # Public list of verified doctors
    path(
        'doctors/',
        views.verified_doctors_list,
        name='doctors_list'
    ),

    # Public doctor profile (ANYONE)
    path(
        'doctors/<int:doctor_id>/',
        views.public_doctor_profile,
        name='public_doctor_profile'
    ),
    path('doctors/<int:doctor_id>/video/', views.start_video_consultation, name='video_consultation'),
    path('doctors/<int:doctor_id>/articles/', views.medical_articles, name='medical_articles'),
    path('doctors/<int:doctor_id>/share/', views.share_profile, name='share_profile'),
    # Doctor features
    path('doctor/appointments/', views.doctor_appointments, name='doctor_appointments'),
    path('send-report/', views.send_report_to_doctor, name='send_report_to_doctor'),
    path('doctor/reports/', views.doctor_pending_reports, name='doctor_pending_reports'),
    path('doctor/reports/<int:report_id>/review/', views.review_report, name='review_report'),
    path('doctor/reports/delete/<int:report_id>/', views.delete_report, name='delete_report'),

    # path('doctor/resources/', views.medical_resources, name='medical_resources'),
    # path('doctor/prescriptions/', views.doctor_prescriptions, name='doctor_prescriptions'),
    
    # Booking
    path(
        'doctors/<int:doctor_id>/book/',
        views.book_appointment,
        name='book_appointment'
    ),
    # Doctor reportswc
path(
    'doctor/reports/',
    views.doctor_reports,
    name='doctor_reports'
),

    # Patient report pages
    path('my_reports/', views.my_reports, name='my_reports'),

    # ----------------------------
    # Admin Pages
    # ----------------------------
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin_dashboard/verify_doctor/<int:doctor_id>/', views.verify_doctor, name='verify_doctor'),
    path('admin_download/download_users_pdf/', views.download_users_pdf, name='download_users_pdf'),

    # ----------------------------
    # Social Auth & Logout
    # ----------------------------
    path('social-complete/', views.social_auth_complete, name='social_auth_complete'),
    path('logout/', views.logout_view, name='logout'),
]
