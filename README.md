# aris-repo-1

## HOW TO START

=======
### Git 로컬 설정

* git 설정 진행

`git config --global user.name "Your Name"`

`git config --global user.email "your.email@example.com"`

* 설정 확인

`git config --list`

=======
### Git 로그인 설정

* SSH 키 생성

`ssh-keygen -t ed25519 -C "your.email@example.com"`

* SSH 공개키 복사

`cat ~/.ssh/id_ed25519.pub`

* 공개키 복사 및 Git 계정에 추가
    * Cat 명령어로 표시되는 키 복사
    * github.com/settings/keys 로 접근
    * New SSH Key
    * 이름 입력 후 공개키 복사 > Add SSH Key

* 키 테스트

`ssh -T git@github.com`

### Repository 클론 및 리모트 설정

* git Clone

`git clone git@github.com:addinedu-advance-3rd/aris-repo-1.git`

* 리모트 설정


=======
`git remote set-url origin git@github.com:addinedu-advance-3rd/aris-repo-1.git`

------
# docker-compose 사용방법
- ## 설치파일
  - [docker 및 docker-compose 설치](https://zhfvkq.tistory.com/41)
  - nvidia-docker 설치
  ~~~
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  ~~~
  ~~~
    sudo apt update
    sudo apt install nvidia-container-toolkit
    ~~~
    ~~~
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  ~~~

- ## 실행방법
  - git에서의 develop branch  
    "feat: init_docker_compose" 기준.
  ~~~
  sudo docker-compose up --build
  ~~~
  - build 는 파일 변경이 있을때 초기 한번만 하면 됨.
  
  - 각자 작성하신 코드내에서 사용하는 포트가 있을거임. 모든 container 가 host와 연결되어 있음. 즉, 
    - face_detect는 (https://0.0.0.0:5000/video)
    - gui는 (https://0.0.0.0:3001)
    - DB는 (https://0.0.0.0:8000)
   - 종료하려면 ctrl + c 를 한번만 누르고 종료되길 기다리는걸 추천. (강제종료는 두번이상 누르면 됨)
  ## 디버깅 방법
  - 각자 폴더에 있는 파일을 수정하고 각자의 머신(docker를 사용하지않고)에서 테스트해보고
    ~~~
    sudo docker-compose up --build
    ~~~  
    로 전체를 다시 실행해도 됨.
  
  - 만약 conatiner 내에서 테스트해보고싶다면 예시로 face_detect 일때.
  src/face_detect 위치에서 
    ~~~
    sudo docker build . --tag test_facedetect
    ~~~
    = test_facedetect의 이름을 갖는 이미지를 만들어줘
    ~~~
    sudo docker run -it --device /dev/video0:/dev/video0 --gpus '"device=0"' --network host test_facedetect /bin/bash
    ~~~
    -it : /bin/bash를 실행하고 입출력 공유를 하겠다.(터미널을 열겠다)  


    --device : /dev/video0:/dev/video0 로써 로컬의 /dev/video0 와 컨테이너 내의 /dev/video0 를 연결 하겠다.  
    
    --gpus : '"device=0"' 로써 0번 gpu를 사용하겠다.  
    test_facedetect의 이미지로 

    --network : host 네트워크를 사용하겠다.  

    test_facedetect : 이 이미지를 사용하겠다.