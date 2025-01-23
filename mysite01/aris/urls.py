from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CusViewSet

from . import views

router = DefaultRouter()
router.register(r'Customers', CusViewSet)

urlpatterns = [
    path('', views.db_update),
    #path('', include(router.urls)),
]
