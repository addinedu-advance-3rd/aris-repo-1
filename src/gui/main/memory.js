
let mediaRecorder;
let recordedChunks = [];
let videoStream;
const videoElement = document.getElementById('video');
const startButton = document.getElementById('start');
const status = document.getElementById('status');
let DEFAULT_DEVICE_ID = 'sdf';
const FALLBACK_DEVICE_ID = '3a44ff7781f8098b3d253d6d6660407fa39dface2eeb2b6f778e01d86140147d';
import { NGROK_BASE_URL } from './config.js';
import { speak } from './tts.js';

document.addEventListener('DOMContentLoaded', () => {
  // 페이지 요소 가져오기
  //const testButton = document.getElementById('test-button');
  const cookingPage = document.getElementById('cooking');
  const mainPage = document.getElementById('main-page');

  // ✅ TTS: 아이스크림 제조 중 안내
  speak("아이스크림을 제조 중입니다. 잠시만 기다려 주세요.");

  // 제조 완료 상태 체크 함수
  function checkEndStatus() {
    fetch(`${NGROK_BASE_URL}/control/check_end_status`)
      .then(response => response.json())
      .then(data => {
        console.log("📡 Manufacturing status:", data);

        if (data.status === "end_ice") {
          // ✅ TTS: 제조 완료 후 안내
          speak("제조가 완료되었습니다! 아이스크림 수령 촬영을 위한 녹화가 진행됩니다.");

          // 제조 완료 상태일 때 페이지 전환
          //페이지 전환 후 캠 띄우기
          setTimeout(() => {
          
            console.log("🚀 캠 활성화 시작");
            console.log("📌 image-container 상태:", document.getElementById('image-container'));
            console.log("📌 cup_detect_stream 상태:", document.getElementById('cup_detect_stream'));
          
          
            console.log("🚀 캠 활성화 시작");
            requestAnimationFrame( () => {
              setTimeout(() => {
                setStreamImage();
              }, 500);
            })
            // 캠 스트림 시작
          }, 500); // 0.5초 후 실행

          clearInterval(statusCheckInterval);   // 주기적 상태 체크 중단
        }
      })
      .catch(error => {
        console.error("Error checking manufacturing status:", error);
      });
    }
    // 상태를 1초마다 주기적으로 체크
  const statusCheckInterval = setInterval(checkEndStatus, 1000);
});


function resetRecordingStatus() {
  fetch(`${NGROK_BASE_URL}/gui/reset_video_recording_status`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-cache'
    }
  })
    .then(response => response.json())
    .then(data => {
      console.log("🔄 Video recording status reset:", data);
      setTimeout(() => {
        fetchRecordingStatus(); // Reset 후 fetch 다시 실행
      }, 1000);
    })
    .catch(error => {
      console.error("⚠️ Error resetting video recording status:", error);
    });

}

// ✅ memory.html 방문 시 실행
document.addEventListener("DOMContentLoaded", function () {
  resetRecordingStatus();
});

// ✅ Fetch the latest video recording status
function fetchRecordingStatus() {
  const timestamp = new Date().getTime();
  fetch(`${NGROK_BASE_URL}/gui/video_recording_status?timestamp=${timestamp}`)
    .then(response => response.json())
    .then(data => {
      console.log("📡 Video recording done status:", data);
      // Update UI accordingly

      if (data.status === "done") {
        if (NGROK_BASE_URL === '') {
          console.log("🔄 Video recording done status:", data);
          window.location.href = '/gui/play.html?video=' + encodeURIComponent(data.url);
        } else {
          console.log("🔄 Video recording done status:", data);
          window.location.href = '/play.html?video=' + encodeURIComponent(data.url); // 토핑 선택 페이지로 이동
        }

        // window.location.href = `/gui/play.html?video=${encodeURIComponent(data.url)}`;
      }
    })
    .catch(error => { 
      console.error("⚠️ Error getting video recording done status:", error);
    });

}

// Call this function periodically or on user interaction
setInterval(fetchRecordingStatus, 1000); // Fetch status every 5 seconds

// 스트리밍 이미지 설정 함수

function setStreamImage() {
  const img_container_element = document.getElementById('image-container');
  let imgElement = document.getElementById('cup_detect_stream'); // 기존 img 태그 가져오기
  let test_p = document.getElementById('test_p');
  // 기존 이미지 태그가 없으면 새로 생성

  const streamURL = `${NGROK_BASE_URL}/cup/video`;

  console.log("✅ 캠 스트림 시작! URL:", streamURL);
  
  imgElement.src = streamURL; // 스트림 URL 설정

  // 이미지 로드 성공/실패 확인
  imgElement.onload = () => {
    console.log("✅ 캠 스트림 로드 완료!");

    console.log("H1 element:", document.querySelector("#main-page h1"));
    console.log("Before removing hidden:", document.getElementById('main-page').classList);
    document.getElementById('main-page').classList.remove('hidden');
    document.getElementById('main-page').classList.add('show-main');
    console.log("After removing hidden:", document.getElementById('main-page').classList);
    console.log("Main page visibility:", window.getComputedStyle(document.getElementById('main-page')).display);
    console.log("Main page opacity:", window.getComputedStyle(document.getElementById('main-page')).opacity);
    console.log("Main page visibility:", window.getComputedStyle(document.getElementById('main-page')).visibility);
    const cookingPage = document.getElementById('cooking');
    const mainPage = document.getElementById('main-page');
  
    cookingPage.classList.add('hidden');  // 랜딩 페이지 숨기기
    mainPage.classList.remove('hidden');  // 메인 페이지 표시




    imgElement.style.display = 'block';
    imgElement.style.opacity = '1';
  }
    
  imgElement.onerror = () => console.error("❌ 캠 스트림 로드 실패! URL을 확인하세요.");
}

//  window.setStreamImage = setStreamImage;

