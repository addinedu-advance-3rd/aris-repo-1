let mediaRecorder;
let recordedChunks = [];
let videoStream;
const videoElement = document.getElementById('video');
const startButton = document.getElementById('start');
const status = document.getElementById('status');
let DEFAULT_DEVICE_ID = 'sdf';
const FALLBACK_DEVICE_ID = '3a44ff7781f8098b3d253d6d6660407fa39dface2eeb2b6f778e01d86140147d';
import { NGROK_BASE_URL } from './config.js';






async function initializeWebcam() {
  
  try {
    // 브라우저가 마이크/카메라 권한을 묻고 허용해야 label 정보를 제대로 확인할 수 있음
    await navigator.mediaDevices.getUserMedia({ video: true, audio: false });

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    videoDevices.forEach(device => {
      console.log(`Label: ${device.label}, DeviceID: ${device.deviceId}`);
      DEFAULT_DEVICE_ID = device.deviceId;
      console.log(DEFAULT_DEVICE_ID);
    });
  } catch (err) {
    console.error('Error enumerating devices:', err);
  }

  try {
    // 모든 장치 가져오기
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    if (videoDevices.length === 0) {
      throw new Error('No video input devices found');
    }

    // 기본값 또는 대체값에 해당하는 deviceId 찾기
    console.log(DEFAULT_DEVICE_ID);
    const selectedDeviceId = videoDevices.find(device => device.deviceId === DEFAULT_DEVICE_ID)?.deviceId
      || videoDevices.find(device => device.deviceId === FALLBACK_DEVICE_ID)?.deviceId;

    if (!selectedDeviceId) {
      throw new Error('Neither default nor fallback device found');
    }

    console.log(`Using device ID: ${selectedDeviceId}`);

    // 웹캠 스트림 초기화
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: selectedDeviceId } },
      audio: false
    });

    videoStream = stream;
    videoElement.srcObject = stream;

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = function(event) {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = function() {
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      recordedChunks = [];
      uploadVideo(blob); // 업로드 및 리디렉션
    };

    status.textContent = `Webcam initialized using device ID: ${selectedDeviceId}`;
  } catch (err) {
    console.error("Error accessing webcam:", err);
    status.textContent = "카메라를 사용할 수 없습니다.";
  }
}

// 촬영 시작 버튼
startButton.addEventListener('click', function() {
  if (!mediaRecorder) {
    status.textContent = "MediaRecorder가 초기화되지 않았습니다.";
    return;
  }
  recordedChunks = [];
  mediaRecorder.start();
  status.textContent = "촬영 중...";
  startButton.disabled = true;

  setTimeout(() => {
    mediaRecorder.stop();
    status.textContent = "촬영 완료! 업로드 중...";
  }, 5000); // 촬영 시간 조정 (5초)
});

// 업로드 및 리디렉션
function uploadVideo(blob) {
  const formData = new FormData();
  formData.append('video', blob, 'recorded_video.webm');

  fetch(`${NGROK_BASE_URL}/gui/upload`, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      console.log("Video uploaded successfully:", data.url);
      // 업로드 후 play.html로 리디렉션
      window.location.href = `/gui/play.html?video=${encodeURIComponent(data.url)}`;
    })
    .catch(error => {
      console.error("Upload failed:", error);
      status.textContent = "업로드 실패!";
    });
}

// 웹캠 초기화 실행
initializeWebcam();
