
let mediaRecorder;
let recordedChunks = [];
let videoStream;
const videoElement = document.getElementById('video');
const startButton = document.getElementById('start');
const status = document.getElementById('status');
let DEFAULT_DEVICE_ID = 'sdf';
const FALLBACK_DEVICE_ID = '3a44ff7781f8098b3d253d6d6660407fa39dface2eeb2b6f778e01d86140147d';
import { NGROK_BASE_URL } from './config.js';


document.addEventListener('DOMContentLoaded', () => {
  // 페이지 요소 가져오기
  //const testButton = document.getElementById('test-button');
  const cookingPage = document.getElementById('cooking');
  const mainPage = document.getElementById('main-page');

  // 상태 체크 함수
  function checkEndStatus() {
    fetch(`${NGROK_BASE_URL}/control/check_end_status`)
      .then(response => response.json())
      .then(data => {
        console.log("📡 Manufacturing status:", data);

        if (data.status === "end_ice") {
          // 제조 완료 상태일 때 페이지 전환
          cookingPage.classList.add('hidden');  // 랜딩 페이지 숨기기
          mainPage.classList.remove('hidden');  // 메인 페이지 표시

          //페이지 전환 후 캠 띄우기
          setTimeout(() => {
            console.log("🚀 캠 활성화 시작");
            setStreamImage();
          }, 500); // 0.5초 후 실행

          clearInterval(statusCheckInterval);   // 주기적 상태 체크 중단
        }
      })
      .catch(error => {
        console.error("Error checking manufacturing status:", error);
      });
    }
    // 상태를 3초마다 주기적으로 체크
  const statusCheckInterval = setInterval(checkEndStatus, 3000);
});



// ✅ Fetch the latest video recording status
function fetchRecordingStatus() {
  fetch(`${NGROK_BASE_URL}/gui/video_recording_status`)
    .then(response => response.json())
    .then(data => {
      console.log("📡 Video recording done status:", data);
      // Update UI accordingly

      if (data.status === "done") {
        if (NGROK_BASE_URL === '') {
          window.location.href = '/gui/play.html?video=' + encodeURIComponent(data.url);
        } else {
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
  const imgElement = document.getElementById('cup_detect_stream');

  if (!imgElement) {
    console.error("🚨 cup_detect_stream 요소를 찾을 수 없습니다! 페이지 로딩 문제?");
    return;
  }

  console.log("✅ 캠 스트림 시작! URL:", `${NGROK_BASE_URL}/cup/video`);
  imgElement.src = `${NGROK_BASE_URL}/cup/video`;

  // 이미지 로드 성공/실패 확인
  imgElement.onload = () => console.log("✅ 캠 스트림 로드 완료!");
  imgElement.onerror = () => console.error("❌ 캠 스트림 로드 실패! URL을 확인하세요.");
}

window.setStreamImage = setStreamImage;

