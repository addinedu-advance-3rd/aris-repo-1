const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const DEFAULT_PORT = 3001; // 기본 포트
let port = DEFAULT_PORT;
let recordingStatus = "not done";
const cors = require('cors');
app.use(cors());
app.use(express.json());
const video_folder = '/app/main/video_src'


// Set up storage for Multer to save files in "img_src" folder
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, 'video_src')); // Save to img_src folder
  },
  filename: function (req, file, cb) {
    cb(null, `recorded_${Date.now()}.webm`); // Unique file name
  }
});

const upload = multer({ storage });

// Serve static files (e.g., index.html, img_src files)
app.use(express.static(path.join(__dirname)));


// Handle video upload`
app.post('/upload', upload.single('video'), (req, res) => {
    console.log('File uploaded:', req.file);
  
    if (!req.file) {
      console.error("No file received!");
      return res.status(400).json({ error: 'No file uploaded' });
    }
  
    const response = {
      message: 'Video uploaded successfully!',
      url: `video_src/${req.file.filename}`
    };
    console.log('Response sent:', response);
  
    res.status(200).json(response);
  });

//     url = 'http://gui_service:3001/video_recording_done'
// payload = {"status": "done"}
// ✅ POST endpoint to receive video recording completion
app.post('/video_recording_done', (req, res) => {
  if (!req.body || !req.body.status) {
    console.error('❌ Invalid request: Missing status field');
    return res.status(400).json({ error: "Invalid request: 'status' field is required." });
  }

  console.log('📌 Video recording done:', req.body);
  recordingStatus = req.body.status; // Update status
  res.status(200).json({ message: '✅ Video recording done received' });
});

// ✅ GET endpoint for frontend to check status
app.get('/video_recording_status', (req, res) => {
  res.json({ status: recordingStatus });
});


// ✅ 상태 초기화 엔드포인트 
app.post('/reset_video_recording_status', (req, res) => {
  recordingStatus = "not done";  // ✅ 상태 초기화
  console.log('🔄 Video recording status reset to "not done"');
  res.status(200).json({ message: '✅ Video recording status reset' });
});


  
// List videos in img_src for the current session
app.get('/latest-video', (req, res) => {
    // const videoFolder = path.join(__dirname, 'video_src');

    fs.readdir(video_folder, (err, files) => {
      if (err) {
        console.error("Error reading video directory:", err);
        return res.status(500).send('Error reading video directory');
      }
  
      // 파일명을 생성 날짜 순으로 정렬하여 가장 최근 파일 반환
      const latestFile = files
        // .filter(file => file.endsWith('.av')) // 비디오 파일만 필터링
        .map(file => ({
          name: file,
          time: fs.statSync(path.join(video_folder, file)).mtime.getTime()
        }))
        .sort((a, b) => b.time - a.time)[0]; // 최신 파일 정렬
  
      if (latestFile) {
        res.status(200).json({ url: `video_src/${latestFile.name}` });
      } else {
        res.status(404).json({ error: 'No video found' });
      }
    });
  });
  

// 서버 시작 함수
function startServer(port) {
    const server = app.listen(port, () => {
      console.log(`Server running at http://localhost:${port}`);
    });
  
    // 포트 충돌 처리
    server.on('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        console.error(`Port ${port} is in use. Trying port ${port + 1}...`);
        port += 1;
        startServer(port); // 다음 포트로 재시작
      } else {
        console.error('Server error:', err);
      }
    });
  }
  
  // 서버 시작
  startServer(port);