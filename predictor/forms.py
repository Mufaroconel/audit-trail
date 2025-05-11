from django import forms

class PredictionForm(forms.Form):
    Amount = forms.FloatField(label="Transaction Amount")
    AnomalyScore = forms.FloatField(label="Anomaly Score")
    SuspiciousFlag = forms.IntegerField(label="Suspicious Flag (0 or 1)")
    AccountBalance = forms.FloatField(label="Account Balance")


class TransactionUploadForm(forms.Form):
    file = forms.FileField(label="Upload Excel File")


