import json
import os
from django.conf import settings
from .models import Customer

def load_json_and_save_to_db():
    json_file_path = os.path.join(settings.BASE_DIR, 'face_metadata.json')

    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)  # JSON 데이터 파싱
        
        # 'Customer' 키에서 리스트를 가져옵니다.
        customers = data.get('Customer', [])
        
        for info in customers:
            # id를 통해 중복 확인 및 생성
            customer, created = Customer.objects.get_or_create(
                id=info['id'],
                defaults={
                    'name': info['name'],
                    'age': info['age'],
                    'gender': info['gender'],
                    'embedding': info['embedding']
                }
            )
            if not created:
                # 이미 존재하는 경우, 필요한 경우 업데이트
                customer.name = info['name']
                customer.age = info['age']
                customer.gender = info['gender']
                customer.embedding = info['embedding']
                customer.save()

    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print("JSON 파싱 오류 발생.")
    except Exception as e:
        print(f"오류 발생: {e}")
