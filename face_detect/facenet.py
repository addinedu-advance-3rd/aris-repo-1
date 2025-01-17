import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import os

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

# 얼굴 이미지에서 임베딩 추출 함수
def extract_embedding(image):
    """이미지에서 얼굴 검출 후 임베딩 추출"""
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        # 얼굴 영역 잘라내기
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            print("Empty face region detected.")
            return None
        # 얼굴 이미지를 Tensor로 변환
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face).unsqueeze(0).to(device)
        # 임베딩 추출
        embedding = facenet(face_tensor).detach().cpu().numpy().flatten()
        return embedding
    else:
        print("No face detected.")
        return None

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
        embedding = extract_embedding(img_rgb)
        if embedding is not None:
            embeddings[file_name] = embedding
    return embeddings

# 기준 이미지 폴더 설정
img_src_folder = 'img_src'  # 기준 얼굴 이미지가 있는 폴더 경로
reference_embeddings = load_embeddings_from_folder(img_src_folder)
if not reference_embeddings:
    print("No valid face embeddings found in the folder.")
    exit()
else:
    print(f"Loaded {len(reference_embeddings)} reference embeddings.")

# 웹캠으로 실시간 얼굴 검출 및 비교
cap = cv2.VideoCapture(0)  # 웹캠 입력
if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

print("Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # BGR → RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 실시간 얼굴 임베딩 추출
    face_embedding = extract_embedding(frame_rgb)
    if face_embedding is not None:
        best_match = None
        min_distance = float('inf')

        # 기준 얼굴들과 비교
        for file_name, ref_embedding in reference_embeddings.items():
            distance = calculate_distance(ref_embedding, face_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = file_name

        # 유사도 기준: 0.7
        if min_distance < 0.7:
            face_id = f"Matching: {best_match}"
            print(f"Matching face detected: {best_match}, Distance: {min_distance:.4f}")
        else:
            face_id = "Unknown Face"
            print("New face detected (no match).")

        # 얼굴 영역 표시
        for box in mtcnn.detect(frame_rgb)[0]:
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, face_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Webcam Face Detection", frame)
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()