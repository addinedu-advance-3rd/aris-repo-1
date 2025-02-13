# 1. Python 이미지를 베이스로 설정
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일을 컨테이너에 복사
COPY requirements.txt /app/

# 4. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt && apt-get update && apt-get install -y curl

# 5. 전체 프로젝트 파일을 복사
COPY . /app/

# 6. 환경 변수 설정
ENV PYTHONUNBUFFERED 1

# 7. 포트 설정
EXPOSE 8000

# 8. Gunicorn을 사용하여 Django 실행
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "config.wsgi:application", "--workers", "3"]
