from django.http import JsonResponse
from .utils import load_json_and_save_to_db

#REST
from rest_framework import viewsets
from .models import Customer
from .serializers import CustomerSerializer

from .fetch_cus import fetch_cus


def db_update(request):
    # load_json_and_save_to_db 함수 호출
    try:
        load_json_and_save_to_db()  # JSON 파일을 로드하고 DB에 저장
        fetch_cus()
        return JsonResponse({'status': 'success', 'message': 'Database updated successfully.'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

# Customer 모델을 위한 뷰셋 클래스 정의
class CustomerViewSet(viewsets.ModelViewSet):
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer