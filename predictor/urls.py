from django.urls import path
from . import views
from .views import CustomAuthToken

urlpatterns = [
    path('', views.home_view, name='home'),  # Homepage view
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    path('predict/', views.predict_view, name='predict'),
    path('import/', views.import_transactions, name='import'),
    path('transactions/', views.transaction_list, name='transaction_list'),
    path('transaction/<int:transaction_id>/<str:status>/', views.update_transaction_status, name='update_transaction_status'),
    path('log-action/', views.log_user_action_view, name='log-action'),
    path('audittrail/', views.view_audittrail_view, name='audittrail'),
    path('client/', views.client_dashboard, name='client'),
    path('api/predict-transaction/', views.transaction_prediction_api, name='transaction_prediction_api'),
    path('api/token/', CustomAuthToken.as_view(), name='api_token_auth'),
    path('api/review-transaction/<str:transaction_id>/', 
         views.admin_review_transaction_api, 
         name='admin_review_transaction_api'),
    path('notifications/', views.notifications_view, name='notifications'),
    path('notifications/mark-read/<int:notification_id>/', 
         views.mark_notification_read, 
         name='mark_notification_read'),
    path('reports/', views.reports_view, name='reports'),
]

