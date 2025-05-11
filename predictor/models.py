from django.db import models
from django.contrib.auth.models import User
from django.utils.timezone import now

class Client(models.Model):
    user_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)
    balance = models.FloatField(default=0.0)
    # Fix the field name clash by adding related_name
    parent_client = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='child_clients')

    def __str__(self):
        return self.name

class Transaction(models.Model):
    TRANSACTION_TYPES = (
        ('deposit', 'Deposit'),
        ('withdraw', 'Withdraw'),
        ('transfer', 'Transfer'),
    )
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Quarantined', 'Quarantined'),
        ('Authorized', 'Authorized'),
        ('Flagged', 'Flagged')
    ]
    
    transaction_id = models.CharField(max_length=100)
    client = models.ForeignKey(Client, on_delete=models.CASCADE, null=True, blank=True)
    transaction_type = models.CharField(max_length=10, choices=TRANSACTION_TYPES, default='deposit')
    amount = models.FloatField()
    recipient = models.ForeignKey(Client, related_name='received_transfers', null=True, blank=True, on_delete=models.SET_NULL)
    is_fraudulent = models.BooleanField(default=False)
    fraud_probability = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(auto_now_add=True)
    anomaly_score = models.FloatField()
    suspicious_flag = models.IntegerField()
    account_balance = models.FloatField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')

    def __str__(self):
        return f"{self.transaction_id} - {self.status}"

class AuditTrail(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(default=now)
    details = models.TextField(blank=True)

    def __str__(self):  # Fix the method name from _str_ to __str__
        return f"{self.timestamp} - {self.user.username} - {self.action}"


class Notification(models.Model):
    PRIORITY_CHOICES = [
        ('high', 'High'),
        ('medium', 'Medium'),
        ('low', 'Low')
    ]
    
    title = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES, default='medium')
    transaction_id = models.CharField(max_length=100)
    fraud_probability = models.FloatField()

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
