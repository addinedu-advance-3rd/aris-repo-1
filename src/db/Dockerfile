# Python 이미지 기반으로 SQLite 설치
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 파일을 컨테이너에 복사
COPY requirements.txt /app/

# pip를 최신 버전으로 업그레이드 후 의존성 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# DB 파일을 저장할 디렉토리 만들기 (기본적으로 /app 디렉토리에 저장되므로 필요에 따라 수정)
RUN mkdir -p /app/db

# 애플리케이션 코드와 DB 파일을 컨테이너에 복사
COPY . /app

# Django 프로젝트를 시작하는 명령어 (이미 Django 프로젝트가 있다면 이 부분은 제외)
# 기본 실행 커맨드 설정: Django 서버 실행
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
