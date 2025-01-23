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
from deepface import DeepFace  # DeepFace 추가


# CUDA 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# 해시된 UUID 생성 함수
def generate_hashed_uuid(length=8):
    full_uuid = str(uuid.uuid4())
    hash_object = hashlib.sha256(full_uuid.encode())
    hashed_uuid = hash_object.hexdigest()[:length]  # 원하는 길이로 자름
    return hashed_uuid
    
# 모델 초기화
mtcnn = MTCNN(keep_all=False, device=device)  # 얼굴 검출용
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # 얼굴 임베딩 추출용

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
def save_new_face_and_embedding(embedding, folder_path, metadata, face_image):
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

    # DeepFace 분석
    try:
        analysis = DeepFace.analyze(face_image, actions=['age', 'gender'], enforce_detection=False)
        if isinstance(analysis, list):  # 분석 결과가 리스트일 경우 처리
            analysis = analysis[0]
        age = analysis.get('age', 'Unknown')

        # 성별 확률 분석
        if isinstance(analysis.get('gender'), dict):  # 성별이 확률로 반환된 경우
            gender_prob = analysis['gender']
            gender = max(gender_prob, key=gender_prob.get)  # 확률이 높은 성별 선택
        else:
            gender = analysis.get('gender', 'Unknown')  # 일반적인 문자열 반환
    except Exception as e:
        print(f"DeepFace 분석 중 오류 발생: {e}")
        age = 'Unknown'
        gender = 'Unknown'

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
img_src_folder = 'img_src'
os.makedirs(img_src_folder, exist_ok=True)

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
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
            if not matched:
                print(f"{matched_name}님 환영합니다. 3초 뒤 종료됩니다.")
                match_start_time = time.time()
                matched = True
                no_match_start_time = None

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{matched_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 3초 후 종료
            if matched and time.time() - match_start_time > 3:
                break  # 루프 탈출 조건 추가
        else:
            matched = False
            if no_match_start_time is None:
                no_match_start_time = time.time()

            if no_match_start_time and time.time() - no_match_start_time > 2:
                print("새로운 얼굴이 감지되었습니다. 임베딩 정보를 저장합니다.")
                save_new_face_and_embedding(embedding, img_src_folder, reference_embeddings, face_image)
                reference_embeddings = load_embeddings_from_folder(img_src_folder)
                no_match_start_time = None

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "No Match", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        print("No face detected.")

    cv2.imshow("Webcam Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
