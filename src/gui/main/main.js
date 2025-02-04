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

export function startFaceDetection() {
const statusMessage = document.getElementById('status');

setInterval(async () => {
    try {
    const response = await fetch(`${NGROK_BASE_URL}/face/check_user`, {
        method: 'GET',
        mode: 'cors',
        headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const result = await response.json();
    const userStatus = result.user_status;
    const recognizedUser = result.recognized_user;

    if (recognizedUser) {
        statusMessage.textContent = `Face detected: ${recognizedUser}`;

        // 사용자 상태에 따라 페이지 이동
        if (userStatus === "new") {
        setTimeout(() => { window.location.href = '/gui/welcome.html'; }, 1000);
        } else if (userStatus === "Recognized") {
        localStorage.setItem('customerNickname', recognizedUser); // 닉네임 저장
        setTimeout(() => { window.location.href = '/gui/topping.html'; }, 1000);
        }
    } else {
        statusMessage.textContent = "No face detected. Please move closer to the camera.";
    }
    } catch (error) {
    console.error("Error checking user:", error);
    }
}, 1000); // 1초마다 상태 확인
}

window.startKiosk = startKiosk;
window.setStreamImage = setStreamImage;
window.startFaceDetection = startFaceDetection;