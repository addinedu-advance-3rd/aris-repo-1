from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CustomerViewSet
from .views import OrderViewSet

router = DefaultRouter()
router.register(r'customers', CustomerViewSet) # 'customers' 엔드포인트에 CustomerViewSet 등록
router.register(r'orders', OrderViewSet) # 'orders' 엔드포인트에 OrderViewSet 등록

urlpatterns = [
    path('', include(router.urls)),
]
