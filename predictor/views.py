from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import joblib
import numpy as np
import pandas as pd
from .models import Transaction, AuditTrail, Notification, Client  # Add Notification here
from .forms import TransactionUploadForm  # Your custom Excel upload form
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.http import JsonResponse
from django.utils import timezone
import datetime
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response

class CustomAuthToken(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data,
                                         context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'token': token.key,
            'user_id': user.pk,
            'email': user.email,
            'username': user.username
        })

model = joblib.load('model.pkl')

# 🔐 User Registration
def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Registration successful.")
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'predictor/register.html', {'form': form})

# 🔐 Login
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Invalid username or password.")
            return render(request, 'predictor/login.html')

    return render(request, 'predictor/login.html')

# 🔐 Logout
def logout_view(request):
    logout(request)
    return redirect('login')

# 🏠 Home Page
@login_required
def home_view(request):
    notifications = Notification.objects.all().order_by('-created_at')[:5]  # Get latest 5 notifications
    unread_count = Notification.objects.filter(is_read=False).count()
    
    return render(request, 'predictor/home.html', {
        'notifications': notifications,
        'unread_count': unread_count
    })

# 🧠 Predict View
@login_required
def predict_view(request):
    if request.method == 'POST':
        try:
            Amount = float(request.POST['Amount'])
            AnomalyScore = float(request.POST['AnomalyScore'])
            SuspiciousFlag = int(request.POST['SuspiciousFlag'])
            AccountBalance = float(request.POST['AccountBalance'])

            features = np.array([[Amount, AnomalyScore, SuspiciousFlag, AccountBalance]])
            prediction = model.predict(features)[0]

            return render(request, 'predictor/result.html', {
                'prediction': prediction
            })
        except Exception as e:
            return render(request, 'predictor/result.html', {
                'prediction': f"Error occurred: {e}"
            })

    return render(request, 'predictor/predict.html')

# 📥 Import Transactions from Excel
@login_required
def import_transactions(request):
    if request.method == 'POST':
        form = TransactionUploadForm(request.POST, request.FILES)
        if form.is_valid():
            df = pd.read_excel(request.FILES['file'])

            for _, row in df.iterrows():
                Transaction.objects.create(
                    transaction_id=row['TransactionID'],
                    amount=row['Amount'],
                    anomaly_score=row['AnomalyScore'],
                    suspicious_flag=row['SuspiciousFlag'],
                    account_balance=row['AccountBalance'],
                    status='Pending'
                )
            
            messages.success(request, "Transactions imported successfully.")
            return redirect('transaction_list')
    else:
        form = TransactionUploadForm()
    return render(request, 'predictor/import.html', {'form': form})

# 📄 List & Manage Transactions
@login_required
def transaction_list(request):
    transactions = Transaction.objects.all()
    return render(request, 'predictor/transactions.html', {'transactions': transactions})

# 🛡️ Quarantine, Authorize, or Flag a transaction
@login_required
def update_transaction_status(request, transaction_id, status):
    transaction = Transaction.objects.get(id=transaction_id)
    transaction.status = status
    transaction.save()
    return redirect('transaction_list')
from django.shortcuts import render
from .models import AuditTrail

def log_user_action_view(request):
    if request.user.is_authenticated:  # Ensure user is logged in
        # Log an action when this view is visited
        AuditTrail.objects.create(
            user=request.user,  # Logs the current logged-in user
            action='Test Action',  # Action description
            details='This is a test action to verify audit logging.'  # Additional details
        )


from .models import AuditTrail

def view_audittrail_view(request):
    logs = AuditTrail.objects.all().order_by('-timestamp')  # Latest first
    return render(request, 'predictor/audittrail.html', {'logs': logs})
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Client, Transaction
import random

def predict_fraud(transaction_data):
    # Dummy fraud logic
    prob = random.random()
    return prob > 0.7, prob

def client_dashboard(request):
    client = Client.objects.get(user_id="user123")  # Static for demo
    if request.method == 'POST':
        t_type = request.POST['transaction_type']
        amount = float(request.POST['amount'])
        recipient_id = request.POST.get('recipient')
        recipient = Client.objects.filter(user_id=recipient_id).first() if recipient_id else None

        # Fraud check
        is_fraud, prob = predict_fraud({
            "type": t_type, "amount": amount, "recipient": recipient_id
        })

        if is_fraud:
            Transaction.objects.create(
                client=client, transaction_type=t_type, amount=amount,
                recipient=recipient, is_fraudulent=True, fraud_probability=prob
            )
            messages.error(request, f"Transaction flagged as FRAUDULENT. Probability: {prob:.2f}")
            return redirect('client_dashboard')

        # Process transaction
        if t_type == 'deposit':
            client.balance += amount
        elif t_type == 'withdraw':
            if client.balance < amount:
                messages.error(request, "Insufficient balance.")
                return redirect('client_dashboard')
            client.balance -= amount
        elif t_type == 'transfer':
            if not recipient:
                messages.error(request, "Recipient not found.")
                return redirect('client_dashboard')
            if client.balance < amount:
                messages.error(request, "Insufficient balance for transfer.")
                return redirect('client_dashboard')
            client.balance -= amount
            recipient.balance += amount
            recipient.save()

        client.save()
        Transaction.objects.create(
            client=client, transaction_type=t_type, amount=amount,
            recipient=recipient, is_fraudulent=False, fraud_probability=prob
        )
        messages.success(request, f"{t_type.capitalize()} successful!")
        return redirect('client_dashboard')

    return render(request, 'dashboard.html', {'client': client})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def transaction_prediction_api(request):
    try:
        # Extract all fields from request
        data = {
            'transaction_id': request.data.get('transaction_id'),
            'account_id': request.data.get('account_id'),
            'transaction_date': request.data.get('transaction_date'),
            'transaction_type': request.data.get('transaction_type'),
            'location': request.data.get('location'),
            'device_id': request.data.get('device_id'),
            'ip_address': request.data.get('ip_address'),
            'login_time': request.data.get('login_time'),
            'amount': float(request.data.get('amount', 0)),
            'anomaly_score': float(request.data.get('anomaly_score', 0)),
            'suspicious_flag': int(request.data.get('suspicious_flag', 0)),
            'account_balance': float(request.data.get('account_balance', 0))
        }

        # Validate required fields
        required_fields = [
            'transaction_id', 'account_id', 'transaction_date', 
            'transaction_type', 'amount', 'account_balance'
        ]
        
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return JsonResponse({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }, status=400)

        # Prepare features for model prediction
        features = np.array([[
            data['amount'],
            data['anomaly_score'],
            data['suspicious_flag'],
            data['account_balance']
        ]])

        # Get prediction and probability
        prediction = model.predict(features)[0]
        fraud_probability = float(model.predict_proba(features)[0][1])

        # Create transaction record
        transaction = Transaction.objects.create(
            transaction_id=data['transaction_id'],
            amount=data['amount'],
            anomaly_score=data['anomaly_score'],
            suspicious_flag=data['suspicious_flag'],
            account_balance=data['account_balance'],
            transaction_type=data['transaction_type'],
            is_fraudulent=prediction,
            fraud_probability=fraud_probability,
            status='Quarantined' if prediction else 'Pending'
        )

        # Create audit log
        AuditTrail.objects.create(
            user=request.user,
            action='Transaction Prediction',
            details=f"""
                Transaction ID: {data['transaction_id']}
                Location: {data['location']}
                Device ID: {data['device_id']}
                IP Address: {data['ip_address']}
                Login Time: {data['login_time']}
                Prediction Result: {'Fraudulent' if prediction else 'Normal'}
                Fraud Probability: {fraud_probability:.2f}
            """
        )

        if prediction:
            # Create notification for fraudulent transaction
            Notification.objects.create(
                title=f"Fraudulent Transaction Detected - {data['transaction_id']}",
                message=f"""
                    Suspicious transaction detected:
                    Amount: ${data['amount']}
                    Location: {data['location']}
                    Device ID: {data['device_id']}
                    IP Address: {data['ip_address']}
                    Fraud Probability: {fraud_probability:.2%}
                """,
                priority='high',
                transaction_id=data['transaction_id'],
                fraud_probability=fraud_probability
            )

            # Create audit trail entry
            AuditTrail.objects.create(
                user=request.user,
                action='Fraudulent Transaction Detection',
                details=f'Transaction {data["transaction_id"]} flagged as fraudulent with {fraud_probability:.2%} probability'
            )

        return JsonResponse({
            'status': 'success',
            'transaction_id': data['transaction_id'],
            'prediction': {
                'is_fraudulent': bool(prediction),
                'fraud_probability': fraud_probability,
                'status': 'Quarantined' if prediction else 'Pending'
            },
            'message': 'Transaction requires review' if prediction else 'Transaction appears safe'
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@login_required
def notifications_view(request):
    notifications = Notification.objects.all().order_by('-created_at')
    unread_count = Notification.objects.filter(is_read=False).count()
    
    return render(request, 'predictor/notifications.html', {
        'notifications': notifications,
        'unread_count': unread_count
    })

# Move this function outside of transaction_prediction_api
@api_view(['POST'])
@permission_classes([IsAdminUser])
def admin_review_transaction_api(request, transaction_id):
    try:
        transaction = Transaction.objects.get(transaction_id=transaction_id)
        new_status = request.data.get('status')
        
        if new_status not in ['Approved', 'Rejected', 'Quarantined']:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid status. Must be Approved, Rejected, or Quarantined'
            }, status=400)

        # Update transaction status
        transaction.status = new_status
        transaction.save()

        # Create audit trail entry
        AuditTrail.objects.create(
            user=request.user,
            action='Transaction Review',
            details=f'Transaction {transaction_id} marked as {new_status}'
        )

        # Create notification for status change
        Notification.objects.create(
            title=f"Transaction Status Updated - {transaction_id}",
            message=f"Transaction has been marked as {new_status} by admin {request.user.username}",
            priority='medium',
            transaction_id=transaction_id,
            fraud_probability=transaction.fraud_probability
        )

        return JsonResponse({
            'status': 'success',
            'transaction_id': transaction_id,
            'new_status': new_status,
            'message': f'Transaction successfully marked as {new_status}'
        })

    except Transaction.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Transaction not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required
def home(request):
    context = {
        # ... your existing context ...
        'notifications': Notification.objects.all().order_by('-created_at')[:5],  # Show latest 5
        'unread_notifications_count': Notification.objects.filter(is_read=False).count()
    }
    return render(request, 'predictor/home.html', context)

@login_required
def mark_notification_read(request, notification_id):
    notification = Notification.objects.get(id=notification_id)
    notification.is_read = True
    notification.save()
    return redirect('notifications')


from django.shortcuts import render
from .models import Transaction
import pandas as pd

@login_required
def reports_view(request):
    # Fetch all transactions
    transactions = Transaction.objects.all()

    # Convert to DataFrame for analysis
    df = pd.DataFrame(list(transactions.values()))

    # Perform analysis
    total_transactions = df.shape[0]
    total_fraudulent = df[df['is_fraudulent']].shape[0]
    total_non_fraudulent = total_transactions - total_fraudulent
    average_amount = df['amount'].mean()
    fraud_probability_avg = df['fraud_probability'].mean()

    # Prepare context for template
    context = {
        'total_transactions': total_transactions,
        'total_fraudulent': total_fraudulent,
        'total_non_fraudulent': total_non_fraudulent,
        'average_amount': average_amount,
        'fraud_probability_avg': fraud_probability_avg,
    }

    return render(request, 'predictor/reports.html', context)

