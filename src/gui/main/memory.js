
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
  setStreamImage();
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


function stopCamera() {
  fetch(`${NGROK_BASE_URL}/cup/stop_camera`, {
    method: 'POST',
  })
  .then(response => response.json())
  .then(data => {
    console.log("🔄 Camera stopped:", data);
  })
  .catch(error => {
    console.error("⚠️ Error stopping camera:", error);
  });
}

window.addEventListener("beforeunload", function () {
  stopCamera();
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
setInterval(fetchRecordingStatus, 2000); // Fetch status every 2 seconds




function setStreamImage() {
  const imgElement = document.getElementById('cup_detect_stream');
  imgElement.src = `${NGROK_BASE_URL}/cup/video`;
  }

window.setStreamImage = setStreamImage;

  


// 웹캠 초기화 실행
initializeWebcam();
