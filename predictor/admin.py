from django.contrib import admin  # Fixed import statement
from .models import Transaction, Client, AuditTrail

# Register your models
admin.site.register(Transaction)
admin.site.register(Client)
admin.site.register(AuditTrail)