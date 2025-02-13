import { NGROK_BASE_URL } from './config.js';
import { speak } from './tts.js';

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

document.addEventListener('DOMContentLoaded', async () => {
    try {
      //TTS
      speak("영상이 재생됩니다.")

      // 최신 영상 가져오기
      const response = await fetch(`${NGROK_BASE_URL}/gui/latest-video`);
      if (!response.ok) {
        throw new Error('Failed to fetch latest video');
      }
  
      const data = await response.json();
      if (!data || !data.url) {
        throw new Error('Invalid video URL');
      }
  
      console.log('Fetched video data:', data);
  
      // URL 파라미터 확인
      const params = new URLSearchParams(window.location.search);
      const autoShare = params.get('invokeShare') === 'true';
  
      const videoElement = document.getElementById('captured-video');
      const qrCodeContainer = document.getElementById('qr-code');
      const playbackButton = document.getElementById('start-playback');
      const fullVideoUrl = `${window.location.origin}${data.url}`;
      const qrCodeUrl = `${window.location.origin}/gui/play.html?video=${encodeURIComponent(data.url)}&invokeShare=true`;
  
      const shareButton = document.getElementById('share-button');
      const backButton = document.getElementById('back');

      console.log('Full video URL:', fullVideoUrl);
      console.log('QR Code URL:', qrCodeUrl);
  
      // 최신 영상 URL 설정
      videoElement.src = data.url;
  
      // 자동 공유 기능 실행
      if (autoShare && navigator.share) {
        try {
          await navigator.share({
            title: 'Captured Video',
            text: 'Check out this video I captured!',
            url: fullVideoUrl, // 실제 공유할 URL
          });
          console.log('Video shared successfully!');
        } catch (err) {
          console.error('Error sharing video:', err);
        }
      }
  
      // QR 코드 생성
      if (qrCodeContainer) {
        try {
          const qrUrl = await QRCode.toDataURL(qrCodeUrl); // QR 코드에 play.html URL 포함
          console.log('Generated QR Code URL:', qrUrl);
  
          // QR 코드 이미지를 DOM에 추가
          const img = document.createElement('img');
          img.src = qrUrl;
          qrCodeContainer.appendChild(img);
        } catch (err) {
          console.error('Error generating QR Code:', err);
        }
      } else {
        console.error('QR Code container not found!');
      }
      


    // 디바이스 확인
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    if (isMobile) {
      shareButton.classList.add('mobile'); // 모바일 디바이스에서만 크게 표시
      backButton.classList.add('mobile'); // 모바일 디바이스에서만 크게 표시
      shareButton.style.display = 'block'; // 버튼 표시
    }

      if (autoShare) {
        document.getElementById('hint').textContent = 'Please click the "Share Video" button to share!';
      }

      shareButton.addEventListener('click', async () => {
        if (navigator.share) {
          try {
            await navigator.share({
              title: 'Captured Video',
              text: 'Check out this video I captured!',
              url: fullVideoUrl,
            });
            console.log('Video shared successfully!');
          } catch (err) {
            console.error('Error sharing video:', err);
          }
        } else {
          alert('Sharing is not supported on this device.');
        }
      });
  
  

      // 재생 버튼 동작
      if (playbackButton && videoElement) {
        playbackButton.addEventListener('click', async () => {
          try {
            await videoElement.play();
            console.log('Video playback started successfully.');
          } catch (err) {
            console.error('Error during video playback:', err);
          }
        });
      } else {
        console.error('Playback button or video element not found.');
      }
    } catch (err) {
      console.error('Error fetching latest video:', err);
      document.body.innerHTML = '<h2>Failed to load the latest video</h2>';
    }
  });
  


  //  time out to redirect 10 se

// setTimeout(() => {
//   window.location.href = '/gui/';
//   console.log('Redirecting to /gui/ in 10 seconds...');
// }, 10000);


const timer = document.getElementById('timer');
let countdown = 15;  // ✅ 원하는 초 단위 설정 (예: 5초) -> 15초로 변경

function updateTimer() {
  timer.textContent = `${countdown}초 후 돌아가기`;  // ✅ 텍스트 업데이트
  countdown--;
  //5초전
  if(countdown == 5){
    speak("잠시 후 초기화면으로 돌아갑니다.")
  }
  if (countdown < 0) {
    clearInterval(timerInterval);  // ✅ 타이머 종료
    window.location.href = "/gui/";  // ✅ 0초가 되면 페이지 이동
  }
}

// ✅ 1초마다 updateTimer 실행
const timerInterval = setInterval(updateTimer, 1000);

// ✅ 초기 실행 (0초 딜레이 방지)
updateTimer();
