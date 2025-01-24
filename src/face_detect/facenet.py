import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import os
import json
import time
import uuid
import hashlib
import requests
from deepface import DeepFace  # DeepFace 추가
import threading  # <-- NEW: for async analysis
from flask import Flask, request, render_template, Response, jsonify
from flask_cors import CORS
from threading import Lock



# -----------------------------
# Global variables for threading
analysis_thread = None
analysis_results = None
new_user_name = None
name_ready = False
# -----------------------------

app = Flask(__name__)
CORS(app)  # 모든 도메인에 대해 CORS 허용\
pending_input = False
pending_face_data = None
face_detected = False  # 얼굴 인지 상태를 관리하는 변수

lock = Lock()  # 상태 업데이트를 위한 락

def set_face_detected(value):
    global face_detected
    with lock:  # 락을 사용하여 스레드 안전하게 업데이트
        face_detected = value

def get_face_detected():
    global face_detected
    with lock:  # 락을 사용하여 스레드 안전하게 읽기
        return face_detected



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
 
# 그대로 유지: DeepFace 분석 함수 (no changes)
def analyze_face_with_deepface(face_image):
    try:
        analysis = DeepFace.analyze(face_image, actions=['age', 'gender'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        age = analysis.get('age', 'Unknown')
        # Check if gender is a dict of probabilities
        if isinstance(analysis.get('gender'), dict):
            gender_prob = analysis['gender']
            gender = max(gender_prob, key=gender_prob.get)
        else:
            gender = analysis.get('gender', 'Unknown')
        return age, gender
    except Exception as e:
        print(f"DeepFace 분석 중 오류 발생: {e}")
        return 'Unknown', 'Unknown'

# NEW: Background thread function to run DeepFace analyze
def analyze_face_in_background(face_image):
    global analysis_results
    # This will run in a separate thread to avoid blocking
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
def save_new_face_and_embedding(embedding, folder_path, metadata, face_image, user_name, analysis_results):
    # user_name = input("이름을 입력하세요: ")  # 사용자로부터 이름 입력
    user_id = generate_hashed_uuid(4)  # 4자리 해시된 UUID 생성
    print(f"user를 새롭게 저장합니다 ~~~~ user_id : {user_id}")
    # JSON 파일 경로
    metadata_path = os.path.join(folder_path, "face_metadata.json")

    # 기존 데이터 불러오기
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            try:
                metadata = json.load(file)
            except json.JSONDecodeError:
                print("Metadata file is corrupted. Reinitializing...")
                metadata = {"Customer": []} #-> 요청한 구조
    else:
        metadata = {"Customer" : [] } # -> json 파일이 없을때 초기화

    # 'Customer" 키 확인 
    if "Customer" not in metadata:
        metadata["Customer"] = [] #없으면 초기화


    # Instead of calling analyze_face_with_deepface here, use the precomputed analysis
    if analysis_results is not None:
        age, gender = analysis_results
    else:
        # fallback if something went wrong in the thread
        age, gender = ('Unknown', 'Unknown')



    # 새로운 데이터 추가
    new_entry = {
        "id": user_id,
        "name": user_name,
        "age": age,
        "gender": gender,
        "embedding": embedding.tolist()
    }
    #리스트에 데이터 추가
    metadata["Customer"].append(new_entry)
    
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
            json.dump({"Customer": []}, file, indent=4, ensure_ascii=False)

    # JSON 파일 읽기
    with open(metadata_path, "r", encoding="utf-8") as file:
        try:
            metadata = json.load(file)
        except json.JSONDecodeError:
            print("Metadata file is corrupted. Reinitializing...")
            metadata = {"Customer": []}  # 기본 구조로 초기화
            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4, ensure_ascii=False)
    # 'Customer' 리스트에서 데이터 로드
    if "Customer" in metadata:
        for customer in metadata["Customer"]:
            user_id = customer["id"]  # 'id'를 키로 사용
            name = customer["name"]  # 사용자 이름
            embedding = np.array(customer["embedding"])  # 임베딩 벡터
            embeddings[user_id] = (name, embedding)

    return embeddings


# 기준 이미지 폴더 설정
img_src_folder = '.'

def gen_frames():
    global analysis_thread
    global analysis_results
    global pending_input
    global pending_face_data
    global new_user_name
    # 메타데이터 로드
    reference_embeddings = load_embeddings_from_folder(img_src_folder)
    if not reference_embeddings:
        print("No valid face embeddings found in the folder.")
    else:
        print(f"Loaded {len(reference_embeddings)} reference embeddings.")

    # 웹캠으로 실시간 얼굴 검출 및 비교
    cap = cv2.VideoCapture(0)

    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Failed to open webcam.")
        exit()

    print("Press 'q' to quit.")
    match_start_time = None  # 매칭된 시간을 저장
    no_match_start_time = None  # 매칭되지 않은 시간을 기록
    matched = False  # 매칭 상태를 저장
    current_user = None # 사용자 변경시 초기화 하는 플래그

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        # BGR -> RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding, boxes, face_image = extract_embedding_and_boxes(frame_rgb)

        if embedding is not None: # 얼굴이 있는 상태
            best_match = "No Match"
            set_face_detected(True)
            min_distance = float('inf')

            # 기준 얼굴들과 비교 # TODO DB 조회 필요
            for file_id, ref_data in reference_embeddings.items():
                ref_name, ref_embedding = ref_data
                distance = calculate_distance(ref_embedding, embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = file_id if min_distance < 0.7 else "No Match"

            if best_match != "No Match":
                matched_name = reference_embeddings[best_match][0]
                matched_key = best_match
                if not matched:
                    print(f"{matched_name}님 환영합니다. 3초 뒤 종료됩니다.",flush=True)
                    current_user = matched_key  #id로 변경완료

                    #print(matched_key) id 잘나오나 확인
                    
                    match_start_time = time.time()
                    matched = True
                    no_match_start_time = None

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{matched_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 3초 후 종료
                if current_user == matched_key:
                    if matched and match_start_time is not None and time.time() - match_start_time > 3:
                        print(f"{matched_name}님 환영합니다. 이제 종료합니다.",flush=True)
                        break  # 루프 탈출 조건 추가
                else:
                    # 현재 사용자와 매칭된 사용자가 달라지면 초기화
                    print(f"사용자가 변경되었습니다: 이전 사용자: {reference_embeddings[current_user][0]}, 새 사용자: {matched_name}")
                    if analysis_thread and analysis_thread.is_alive():
                        print("Stopping background analysis because user changed.",flush=True)
                        # There's no direct 'stop' in Python threads, but we can ignore results.
                        # In a more robust design, you'd have a shared variable to indicate "stop".
                    current_user = None
                    match_start_time = None
                    matched = False        

            else: # best_match = "No Match"
                # NEW FACE DETECTED: start background analysis immediately (if not running)
                if (analysis_thread is None or not analysis_thread.is_alive()):
                    print("New face detected: Starting DeepFace analysis in background...",flush=True)
                    # Reset analysis_results so we know we have fresh data
                    analysis_results = None
                    analysis_thread = threading.Thread(target=analyze_face_in_background,
                                                    args=(face_image,))
                    analysis_thread.start()

                matched = False # 매칭이 실패한 경우 상태 초기화
                if no_match_start_time is None:
                    no_match_start_time = time.time()
                
                # 매칭되지 않고 2초 이상 경과한 경우
                if no_match_start_time and time.time() - no_match_start_time > 2:
                
                    print("새로운 얼굴이 감지되었습니다. 임베딩 정보를 저장합니다.",flush=True)
                    pending_input = True

                    # 얼굴 재확인
                    # embedding, boxes, face_image= extract_embedding_and_boxes(frame_rgb)
                    pending_face_data = extract_embedding_and_boxes(frame_rgb)
                    if embedding is not None and boxes is not None and len(boxes) > 0:
                        print("얼굴이 확인되었습니다. 이미지를 저장합니다.")
                        if analysis_thread is not None and analysis_thread.is_alive():
                            print("Waiting for DeepFace analysis to finish...")
                            # You can block indefinitely or use a timeout
                            analysis_thread.join()


                        save_new_face_and_embedding(embedding, img_src_folder, reference_embeddings, face_image, new_user_name, analysis_results)

                        reference_embeddings = load_embeddings_from_folder(img_src_folder)
                        no_match_start_time = None
                    else:
                        print("얼굴이 사라졌습니다. NO MATCH ")
                        no_match_start_time = None
                    no_match_start_time = None
                elif embedding is not None and boxes is not None and len(boxes) > 0:
                    # 얼굴이 다시 확인되면 NO MATCH 상태 초기화
                    print("얼굴이 다시 확인되었습니다.")
                    # 맞나..? 
                    #no_match_start_time = None

                # 매칭되지 않은 경우 바운딩 박스 표시
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "No Match", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            print("No face detected.")
            no_match_start_time = None
            match_start_time = None
            if analysis_thread and analysis_thread.is_alive():
                print("Stopping background analysis because face is gone.")
                # In practice, you'd have a mechanism to kill or ignore the thread.

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit_name', methods=['POST'])
def submit_name():
    global pending_input
    global pending_face_data
    global analysis_results
    global new_user_name
    global pending_name

    print('submit name started', flush=True)
    print('submit name started', flush=True)
    print('submit name started', flush=True)
    print('submit name started', flush=True)

    if not pending_input :
        return jsonify({"error": "No pending input"}), 400

    if not pending_input or not pending_face_data:
        return jsonify({"error": "No face pending for input"}), 400

    print('embedding,box,face_image', flush=True)
    embedding, boxes, face_image = pending_face_data

    print(len(embedding), flush = True)
    print(len(boxes), flush = True)
    print(len(face_image), flush = True)
    data = request.json
    new_user_name = data.get('name')  # 클라이언트에서 전달된 닉네임
    name_ready = True
    if not new_user_name:
        return jsonify({"error": "Name is required"}), 400
    

    # 분석 결과 확인
    if analysis_results is None:
        return jsonify({"error": "Analysis results not ready"}), 400



    # 상태 초기화
    pending_input = False
    pending_face_data = None
    analysis_results = None 
    return jsonify({"message": f"Face saved for {new_user_name}"}), 200


@app.route('/check_face', methods=['GET'])
def check_face():
    global face_detected
    return jsonify({"face_detected": get_face_detected()})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 5000, threaded=True)