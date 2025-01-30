from django.db import models

# Create your models here.
class Customer(models.Model):
    id = models.CharField(max_length=4, primary_key=True)
    name = models.CharField(max_length=20)
    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=5)
    embedding = models.JSONField()

    def __str__(self):
        return self.id
    