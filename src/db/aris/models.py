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
    
class Order(models.Model):
    order_num = models.CharField(max_length=10, primary_key=True)
    customer_id = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='orders')
    flavor = models.CharField(max_length=10)
    topping = models.JSONField()
    datetime = models.DateTimeField()
    
    def __str__(self):
        return self.order_num