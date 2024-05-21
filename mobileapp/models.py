from django.db import models

# Create your models here.

from unittest.util import _MAX_LENGTH
import os

# Create your models here.

class Register(models.Model):
    name=models.CharField(max_length=50,null=True)
    email=models.EmailField(max_length=50,null=True)
    password=models.CharField(max_length=50,null=True)
    age=models.CharField(max_length=50,null=True)
    contact=models.CharField(max_length=50,null=True)
   




