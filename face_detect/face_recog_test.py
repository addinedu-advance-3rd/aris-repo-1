import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import os

# cuda 사용 여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# MTCNN 모델 초기화 (얼굴 검출용)
mtcnn = MTCNN(keep_all=False, device=device)
# FaceNet 모델 초기화 (얼굴 임베딩 추출용)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 경로 설정
base_path = '/home/addinedu/Downloads/face/'
image_files = ["image1.jpeg", "image2.jpeg", "image3.jpg"]

# 얼굴 비교 함수 (유클리드 거리 계산)
def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# 이미지 파일에서 얼굴 임베딩 추출
face_database = {}
for image_file in image_files:
    image_path = os.path.join(base_path, image_file)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        continue

    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
            face = img[y1:y2, x1:x2]

            if face.size == 0:
                continue

            transform = transforms.ToTensor()
            face_tensor = transform(face).unsqueeze(0).to(device)

            try:
                face_embedding = facenet(face_tensor).detach().cpu().numpy().flatten()
            except Exception as e:  # 여기서 except를 정확히 씁니다.
                print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
                print(f"Face size: {face.shape}")
                print(e)



            face_database[image_file] = face_embedding
            print(f"Stored embedding for {image_file}")

if not face_database:
    print("No valid faces found in the provided images.")
    exit()

# 실시간 비디오 캡처
cap = cv2.VideoCapture(0)
output_width = 640
output_height = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (output_width, output_height))
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                # 얼굴 이미지를 Tensor로 변환 및 크기 조정
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((160, 160))  # InceptionResNetV1의 입력 크기에 맞게 조정
                ])
                face_tensor = transform(face).unsqueeze(0).to(device)

            if face.size == 0:
                continue

            transform = transforms.ToTensor()
            face_tensor = transform(face).unsqueeze(0).to(device)
            face_embedding = facenet(face_tensor).detach().cpu().numpy().flatten()

            matched_face = None
            min_distance = float('inf')

            for name, db_embedding in face_database.items():
                distance = calculate_distance(face_embedding, db_embedding)
                if distance < 0.6 and distance < min_distance:  # 유사도 기준
                    matched_face = name
                    min_distance = distance

            if matched_face:
                print(f"Matched with {matched_face}, Distance: {min_distance}")
                cv2.putText(frame, matched_face, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("No match found")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Real-time Face Comparison", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()