from django.urls import path
from . import views

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

]

