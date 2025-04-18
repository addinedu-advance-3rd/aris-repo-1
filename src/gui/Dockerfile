# Step 1: Node.js 공식 이미지 사용
# 여기서 '14'는 Node.js 버전입니다. 필요에 따라 다른 버전을 선택할 수 있습니다.
FROM node:14

# Step 2: 작업 디렉토리 설정
# /app 디렉토리로 설정
WORKDIR /app

# Step 3: 의존성 파일 복사
# package.json과 package-lock.json 파일을 복사하여 의존성 설치에 사용합니다.
COPY package*.json ./

# Step 4: 의존성 설치
# 'npm install' 명령어로 의존성 설치
RUN npm install

# Step 5: 애플리케이션 코드 복사
# 애플리케이션의 소스 코드를 컨테이너의 /app 디렉토리로 복사합니다.
COPY . /app/

# Step 6: 앱 실행
# 애플리케이션을 실행할 명령어를 설정합니다. 여기서는 'npm start'를 사용합니다.
WORKDIR /app/main
CMD ["node", "server.js"]

# Step 7: 컨테이너가 열어야 할 포트 지정
# Node.js 앱이 실행되는 포트입니다. 예를 들어 3000번 포트.
# sudo docker run --network host -it  507c10e0d63f /bin/bash
