import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import os
import json
import uuid
import hashlib
import threading
from deepface import DeepFace
from flask import Flask, render_template, Response

# -----------------------------
# Global variables for threading
analysis_thread = None
analysis_results = None
# -----------------------------

# Flask 애플리케이션 생성
app = Flask(__name__)

# CUDA 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# 모델 초기화
mtcnn = MTCNN(keep_all=False, device=device)  # 얼굴 검출용
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # 얼굴 임베딩 추출용

# 해시된 UUID 생성 함수
def generate_hashed_uuid(length=8):
    full_uuid = str(uuid.uuid4())
    hash_object = hashlib.sha256(full_uuid.encode())
    hashed_uuid = hash_object.hexdigest()[:length]  # 원하는 길이로 자름
    return hashed_uuid
    
# DeepFace 분석 함수
def analyze_face_with_deepface(face_image):
    try:
        analysis = DeepFace.analyze(face_image, actions=['age', 'gender'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        age = analysis.get('age', 'Unknown')
        if isinstance(analysis.get('gender'), dict):
            gender_prob = analysis['gender']
            gender = max(gender_prob, key=gender_prob.get)
        else:
            gender = analysis.get('gender', 'Unknown')
        return age, gender
    except Exception as e:
        print(f"DeepFace 분석 중 오류 발생: {e}")
        return 'Unknown', 'Unknown'

# 백그라운드에서 DeepFace 분석을 실행하는 함수
def analyze_face_in_background(face_image):
    global analysis_results
    results = analyze_face_with_deepface(face_image)
    analysis_results = results
    print("DeepFace analysis done in background:", results)

# 얼굴 비교 함수 (유클리드 거리 계산)
def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# 얼굴 임베딩 및 바운딩 박스 추출 함수
def extract_embedding_and_boxes(image):
    boxes, _ = mtcnn.detect(image)
    if boxes is not None and len(boxes) > 0:  # 얼굴이 검출된 경우
        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            print("Empty face region detected.")
            return None, None, None
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face).unsqueeze(0).to(device)
        embedding = facenet(face_tensor).detach().cpu().numpy().flatten()
        return embedding, [boxes[0]], face  # 얼굴 이미지 추가 반환
    else:
        print("No face detected.")
        return None, None, None

# 새로운 얼굴의 임베딩 및 메타데이터 저장
def save_new_face_and_embedding(embedding, folder_path, metadata, face_image, analysis_results):
    user_name = input("이름을 입력하세요: ")  # 사용자로부터 이름 입력
    user_id = generate_hashed_uuid(4)  # 4자리 해시된 UUID 생성

    # JSON 파일 경로
    metadata_path = os.path.join(folder_path, "face_metadata.json")

    # 기존 데이터 불러오기
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            try:
                metadata = json.load(file)
            except json.JSONDecodeError:
                print("Metadata file is corrupted. Reinitializing...")
                metadata = {}

    # Instead of calling analyze_face_with_deepface here, use the precomputed analysis
    if analysis_results is not None:
        age, gender = analysis_results
    else:
        # fallback if something went wrong in the thread
        age, gender = ('Unknown', 'Unknown')

    # 새로운 데이터 추가
    metadata[user_id] = {
        "name": user_name,
        "age": age,
        "gender": gender,
        "embedding": embedding.tolist()
    }

    # JSON 파일에 저장
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print(f"{user_name}님의 임베딩 정보와 분석 결과가 저장되었습니다.")

# 폴더 내 모든 이미지 로드 및 임베딩 추출
def load_embeddings_from_folder(folder_path):
    embeddings = {}
    metadata_path = os.path.join(folder_path, "face_metadata.json")

    # JSON 파일 없으면 생성
    if not os.path.exists(metadata_path):
        print("Metadata file not found. Creating a new one...")
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump({}, file, indent=4, ensure_ascii=False)

    # JSON 파일 읽기
    with open(metadata_path, "r", encoding="utf-8") as file:
        try:
            metadata = json.load(file)
        except json.JSONDecodeError:
            print("Metadata file is corrupted. Reinitializing...")
            metadata = {}
            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4, ensure_ascii=False)

    # 임베딩 데이터 로드
    for id, data in metadata.items():
        embeddings[id] = (data['name'], np.array(data['embedding']))

    return embeddings

# 기준 이미지 폴더 설정
img_src_folder = '.'

# 메타데이터 로드
reference_embeddings = load_embeddings_from_folder(img_src_folder)
if not reference_embeddings:
    print("No valid face embeddings found in the folder.")
else:
    print(f"Loaded {len(reference_embeddings)} reference embeddings.")

# OpenCV로 웹캠에서 실시간으로 영상 스트리밍
def gen_frames():
    analysis_thread = None
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Failed to open webcam.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding, boxes, face_image = extract_embedding_and_boxes(frame_rgb)

        if embedding is not None:
            best_match = "No Match"
            min_distance = float('inf')

            # 기준 얼굴들과 비교
            for file_id, ref_data in reference_embeddings.items():
                ref_name, ref_embedding = ref_data
                distance = calculate_distance(ref_embedding, embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = file_id if min_distance < 0.7 else "No Match"

            if best_match != "No Match":
                matched_name = reference_embeddings[best_match][0]
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{matched_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:
                # No match case
                if (analysis_thread is None or not analysis_thread.is_alive()):
                    print("New face detected: Starting DeepFace analysis in background...")
                    analysis_results = None
                    analysis_thread = threading.Thread(target=analyze_face_in_background, args=(face_image,))
                    analysis_thread.start()

                matched = False
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "No Match", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

# Flask 라우팅 설정
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True)
