from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CustomerViewSet

from . import views

router = DefaultRouter()
router.register(r'customers', CustomerViewSet) # 'customers' 엔드포인트에 CustomerViewSet 등록

urlpatterns = [
    path('', views.db_update),
    path('', include(router.urls)),
]
