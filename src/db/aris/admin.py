from django.contrib import admin
from .models import Customer, Order

@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'age', 'gender']
    search_fields = ['name'] # 이름으로 검색

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['order_num', 'customer_id', 'datetime']
    search_fields = ['customer_id__id'] # 고객 id 검색하면 해당 고객의 모든 주문내역
