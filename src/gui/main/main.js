import { NGROK_BASE_URL } from './config.js';

// 배경 컨테이너에 아이템 자동 생성
document.addEventListener('DOMContentLoaded', function() {
    const bgContainer = document.querySelector('.bg-container');
    const rowCount = 22;
    const itemsPerRow = 22;
    const horizontalSpacing = 150; // px
    const verticalSpacing = 100;   // px
    
    let html = '';
    for (let row = 0; row < rowCount; row++) {
      for (let col = 0; col < itemsPerRow; col++) {
        const left = col * horizontalSpacing;
        const top = row * verticalSpacing;
        const imageIndex = (col % 3) + 1; // 순차적으로 1, 2, 3번 이미지 사용
        html += `
          <div class="item" style="top: ${top - 500}px; left: ${left - 700}px;">
            <img src="img_src/ice_img_${imageIndex}.png" alt="Ice Cream">
          </div>
        `;
      }
    }
    bgContainer.innerHTML = html;
  });


// face detection 로직을 실행하는 함수
export function setStreamImage() {
  const imgElement = document.getElementById('faceStream');
  imgElement.src = `${NGROK_BASE_URL}/face/video`;
  }
  
  let interval;
  let isRedirecting = false;
  
  export function startFaceDetection() {
  const statusMessage = document.getElementById('status');
  
  interval = setInterval(async () => {
      if (isRedirecting) return; // 이미 전환중이면 요청중단
  
      try {
        const response = await fetch(`${NGROK_BASE_URL}/face/check_user`, {
          method: 'GET',
          mode: 'cors',
          headers: { 'Content-Type': 'application/json' },
      });
  
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  
      const result = await response.json();
      //const userStatus = result.user_status;
      const recognizedUser = result.recognized_user;
      //여기부터 추가
      const completedStatus = result.completed_status;  // 종료 상태 확인
      const age = result.age;
      const gender = result.gender;
      /// 추가 부 끝
  
      // 종료 상태에 따라 페이지 전환
      if(completedStatus === "in_progress") {  // 캠이 돌고 있을 때
        statusMessage.textContent = "Face detection in progress..";
      }
      else if (completedStatus === "matched") {
        console.log("User matched. Redirecting...");
        //로컬에 이름,나이,성별 저장
        localStorage.setItem('customerNickname', recognizedUser); 
        localStorage.setItem('customerAge', age);
        localStorage.setItem('customerGender', gender);
        
        isRedirecting = true; // 전환 시작 플래그
  
        setTimeout(() => { 
                      window.location.href = '/gui/topping.html';
                  }, 1000);  // 1초 후 페이지 전환
    } else if (completedStatus === "no_match") {
        console.log("No face match. Redirecting...");
  
        isRedirecting = true;  // 전환 시작 플래그 설정
        setTimeout(() => { 
            window.location.href = '/gui/welcome.html';
        }, 1000);  // 1초 후 페이지 전환
    }
  
    // 상태 메시지 업데이트
    if (recognizedUser) {
        statusMessage.textContent = `Face detected: ${recognizedUser}`;
    } else {
        statusMessage.textContent = "No face detected. Please move closer to the camera.";
    }
  
  } catch (error) {
    console.error("Error checking user:", error);
  }
  }, 1000);
  }

window.startKiosk = startKiosk;
window.setStreamImage = setStreamImage;
window.startFaceDetection = startFaceDetection;