from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
import json
from .models import Customer
from .models import Order
from .serializers import CustomerSerializer
from .serializers import OrderSerializer

# Customer 모델을 위한 뷰셋 클래스 정의
class CustomerViewSet(viewsets.ModelViewSet):
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer

    @action(detail=False, methods=['post'])
    def update_from_json(self, request):
    # JSON 파일을 받아서 처리
        try:
            json_data = request.data  # JSON 데이터 가져오기
            # JSON 데이터 처리 및 DB 업데이트
            for item in json_data:
            # ID로 객체를 찾기
                obj = Customer.objects.filter(id=item['id']).first()
                if obj:
                # 객체가 존재하면 업데이트
                    for key, value in item.items():
                        setattr(obj, key, value)
                    obj.save()  # 변경사항 저장
                else:
                # 객체가 존재하지 않으면 새로 생성
                    Customer.objects.create(**item)
            return Response({"status": "success"}, status=status.HTTP_200_OK)
        except json.JSONDecodeError:
            return Response({"error": "Invalid JSON"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
# order 모델을 위한 뷰셋 클래스 정의
class OrderViewSet(viewsets.ModelViewSet):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer

    @action(detail=False, methods=['get'], url_path='customer-orders/(?P<customer_id>[^/.]+)')
    def customer_orders(self, request, customer_id=None):
    #특정 고객의 모든 주문 내역을 조회
        try:
            customer = Customer.objects.get(id=customer_id)  # 고객 찾기
            orders = customer.orders.all()  # 고객의 주문 가져오기
            serializer = self.get_serializer(orders, many=True)
            return Response(serializer.data, status=200)
        except Customer.DoesNotExist:
            return Response({"error": "해당 고객을 찾을 수 없습니다."}, status=404)