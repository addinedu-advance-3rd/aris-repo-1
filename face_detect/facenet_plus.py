import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import os
import time
import json

import threading

img_src_folder = 'img_src'
img_src_test_folder = 'img_src_test'

# 대상 폴더 확인
image_folder = img_src_test_folder

# CUDA 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# 모델 초기화
mtcnn = MTCNN(keep_all=False, device=device)  # 얼굴 검출용
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # 얼굴 임베딩 추출용

# 얼굴 비교 함수 (유클리드 거리 계산)
def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def extract_embedding_and_boxes(image):
    """이미지에서 얼굴 검출 후 첫 번째 얼굴의 임베딩과 바운딩 박스를 반환"""
    boxes, _ = mtcnn.detect(image)
    if boxes is not None and len(boxes) > 0:  # 얼굴이 하나 이상 검출된 경우
        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            print("Empty face region detected.")
            return None, None
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face).unsqueeze(0).to(device)
        embedding = facenet(face_tensor).detach().cpu().numpy().flatten()
        return embedding, [boxes[0]]  # 첫 번째 바운딩 박스만 반환
    else:
        print("No face detected.")
        return None, None

def save_new_face(image, folder_path):
import cv2
import os
import threading
import json
from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=False)  # MTCNN 초기화

def save_new_face(image, folder_path):
    """새로운 얼굴 이미지를 저장하고 나이, 성별을 JSON 파일로 저장"""
    file_name = "random_face.jpg"  # 기본 랜덤 이름
    input_received = threading.Event()

    def get_user_input():
        nonlocal file_name
        name = input("이름을 입력하세요: ")
        if name.strip():
            file_name = name + ".jpg"
        input_received.set()

    print("10초 안에 이름을 입력하세요...")
    input_thread = threading.Thread(target=get_user_input)
    input_thread.start()
    input_thread.join(timeout=10)  # 10초 대기

    if not input_received.is_set():
        print("시간 초과! 랜덤 닉네임으로 저장합니다.")

    file_path = os.path.join(folder_path, file_name)

    # 얼굴 검출 및 바운딩 박스 추출
    boxes, _ = mtcnn.detect(image)
    if boxes is not None and len(boxes) > 0:
        # 첫 번째 얼굴 바운딩 박스만 사용
        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        face = image[y1:y2, x1:x2]  # 얼굴 영역 추출

        if face.size > 0:
            # 얼굴 이미지 저장
            cv2.imwrite(file_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            print(f"얼굴이 저장되었습니다: {file_name}")

            # DeepFace로 나이, 성별 분석
            try:
                from deepface import DeepFace
                analyze_face = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)
                if isinstance(analyze_face, list):  # DeepFace가 리스트로 반환하는 경우 처리
                    analyze_face = analyze_face[0]
                age = analyze_face.get("age", "Unknown")
                gender = analyze_face.get("dominant_gender", "Unknown")
            except Exception as e:
                print(f"DeepFace 분석 중 오류 발생: {e}")
                age = "Unknown"
                gender = "Unknown"

            # JSON 파일에 나이와 성별 저장
            metadata_path = os.path.join(folder_path, "face_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as file:
                    metadata = json.load(file)
            else:
                metadata = {}

            metadata[file_name] = {"age": age, "gender": gender}

            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4, ensure_ascii=False)
            print(f"나이와 성별 정보가 저장되었습니다: {metadata[file_name]}")
        else:
            print("얼굴 영역이 비어 있습니다. 저장하지 않습니다.")
    else:
        print("얼굴이 감지되지 않았습니다. 저장하지 않습니다.")


# 폴더 내 모든 이미지 로드 및 임베딩 추출
def load_embeddings_from_folder(folder_path):
    embeddings = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to load image from {file_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding, boxes = extract_embedding_and_boxes(img_rgb)
        if embedding is not None:
            embeddings[file_name] = embedding
    return embeddings

# 기준 이미지 폴더 설정
reference_embeddings = load_embeddings_from_folder(image_folder)
metadata_path = os.path.join(image_folder, "face_metadata.json")

# 메타데이터 로드
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
else:
    metadata = {}
    print("No metadata found. Please ensure the face_metadata.json file exists.")

if not reference_embeddings:
    print("No valid face embeddings found in the folder.")
    
else:
    print(f"Loaded {len(reference_embeddings)} reference embeddings.")

# 웹캠으로 실시간 얼굴 검출 및 비교
cap = cv2.VideoCapture(0)

# 해상도 설정
width = 600   
height = 480  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

print("Press 'q' to quit.")

match_start_time = None  # 매칭된 시간을 저장
no_match_start_time = None  # 매칭되지 않은 시간을 기록
matched = False  # 매칭 상태를 저장
detection_staus = False
current_user = None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break
    # BGR -> RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 실시간 얼굴 임베딩 추출
    embedding, boxes = extract_embedding_and_boxes(frame_rgb)

    if embedding is not None: #얼굴이 있긴할때
        detection_staus = True
        best_match = "No Match"
        min_distance = float('inf')

        # 기준 얼굴들과 비교
        for file_name, ref_embedding in reference_embeddings.items():
            distance = calculate_distance(ref_embedding, embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = file_name if min_distance < 0.7 else "No Match"

        if best_match != "No Match":
            if not matched:  # 처음 매칭된 경우만 시간 기록
                age = metadata.get(best_match, {}).get("age", "Unknown")
                gender = metadata.get(best_match, {}).get("gender", "Unknown")
                print(f"{best_match}님 환영합니다. 1.5초 뒤 종료됩니다.")

                match_start_time = time.time()
                matched = True
                no_match_start_time = None  # 매칭되면 비매칭 시간 초기화

            # 바운딩 박스와 이름 표시
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if current_user == best_match:
                # 1.5초 후 종료
                if matched and time.time() - match_start_time > 1.5:
                    print(f"{best_match}님 환영합니다. 이제 종료합니다.")
                    break
            else:
                # 현재 사용자와 매칭된 사용자가 달라지면 초기화
                print(f"사용자가 변경되었습니다: 이전 사용자: {current_user}, 새 사용자: {best_match}")
                current_user = best_match  # 새로운 사용자를 설정
                match_start_time = time.time()  # 새로 매칭된 사용자에 대해 타이머 초기화
                matched = True

            matched = False  # 매칭이 실패한 경우 상태 초기화

            # NO MATCH 상태를 처음 기록
            if no_match_start_time is None:
                no_match_start_time = time.time()

            # 매칭되지 않고 2초 이상 경과한 경우
            if no_match_start_time and time.time() - no_match_start_time > 2:
                print("새로운 얼굴이 감지되었습니다. 이미지를 저장하려고 시도합니다.")

                # 얼굴 재확인
                embedding, boxes = extract_embedding_and_boxes(frame_rgb)
                if embedding is not None and boxes is not None and len(boxes) > 0:
                    print("얼굴이 확인되었습니다. 이미지를 저장합니다.")
                    save_new_face(frame_rgb, image_folder)
                    reference_embeddings = load_embeddings_from_folder(image_folder)  # 새로 로드
                else:
                    print("얼굴이 사라졌습니다. NO MATCH ")
                    no_match_start_time = None
                
                # NO MATCH 상태 초기화
                no_match_start_time = None
            elif embedding is not None and boxes is not None and len(boxes) > 0:
                # 얼굴이 다시 확인되면 NO MATCH 상태 초기화
                print("얼굴이 다시 확인되었습니다.")
                

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

    # 화면 출력
    cv2.imshow("Webcam Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
