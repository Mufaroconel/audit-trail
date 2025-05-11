from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import joblib
import numpy as np
import pandas as pd
from .models import Transaction  # Your Transaction model
from .forms import TransactionUploadForm  # Your custom Excel upload form

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
    return render(request, 'predictor/home.html')

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

