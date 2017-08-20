from django import forms
from django.contrib.auth.models import User
from .models import Upload

class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

class DocumentForm(forms.ModelForm):

    class Meta:
        model = Upload
        fields = ['docfile']

    #dfile = forms.FileField(label='Select a file')
