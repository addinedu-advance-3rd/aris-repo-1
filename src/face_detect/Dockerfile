# Python 3.10 slim 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 추가 패키지 설치 (wget 포함)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 wget && \
    rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치 (최적화)
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 모델 파일 다운로드 및 저장 (최적화)
RUN mkdir -p /root/.deepface/weights && \
    wget -O /root/.deepface/weights/gender_model_weights.h5 \
    https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5 && \
    wget -O /root/.deepface/weights/age_model_weights.h5 \
    https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5

# 소스 코드 복사 (최적화)
COPY . /app/

# 애플리케이션 실행
CMD ["python", "new_facenet.py"]
