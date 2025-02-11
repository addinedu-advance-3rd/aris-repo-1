# Python 3.10 slim 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt /app/

# pip를 최신 버전으로 업그레이드 후 의존성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git


# # 2️⃣ YOLOv5 다운로드 (기존 캐시 제거 후 강제 다운로드)
# RUN git clone --depth 1 https://github.com/ultralytics/yolov5.git /app/yolov5
#     # 2️⃣ YOLOv5 패키지 설치 (pip install .)

# # 2️⃣ YOLOv5 디렉토리 확인 (디버깅)
# RUN echo "🔍 YOLOv5 디렉토리 구조:" && ls -l /app/yolov5 && \
#     echo "🔍 현재 작업 디렉토리:" && pwd


# # 3️⃣ YOLOv5 폴더 확인 (디버깅용)
# RUN ls -l /app/yolov5 || echo "🚨 YOLOv5 다운로드 실패"


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# RUN pip list | grep yolov5 || echo "YOLOv5 설치 안됨"
# RUN python -c "import yolov5" || echo "YOLOv5 import 실패"
    


# 필요한 추가 패키지 설치

# RUN apt-get update -y && \
#     apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
#     rm -rf /var/lib/apt/lists/*

# 소스 코드 복사
COPY . /app/

# 애플리케이션 실행
CMD ["python", "test.py"]

# sudo docker run --network host -it  507c10e0d63f /bin/bash
