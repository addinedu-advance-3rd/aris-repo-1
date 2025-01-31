# Python 이미지 기반으로 SQLite 설치
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 라이브러리 설치 (예: sqlite3)
RUN pip install --upgrade pip
RUN pip install flask
RUN pip install numpy

# DB 파일을 저장할 디렉토리 만들기
RUN mkdir -p /data/db

# 애플리케이션 코드와 DB 파일을 컨테이너에 복사
COPY . /app

# 기본 실행 커맨드 설정 (필요한 경우)
CMD ["python", "app.py"]
