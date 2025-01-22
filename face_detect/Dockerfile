# Python 3.10 slim 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt /app/

# pip를 최신 버전으로 업그레이드 후 의존성 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 필요한 추가 패키지 설치
RUN apt-get update -y && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 소스 코드 복사
COPY . /app/

# 애플리케이션 실행
CMD ["python", "facenet.py"]

# sudo docker run --gpus '"device=0"' [container_name]
