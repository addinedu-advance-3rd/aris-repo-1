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
import pickle
import sqlite3
from deepface import DeepFace
import threading
from flask import Flask, request, render_template, Response, jsonify
from flask_cors import CORS
from threading import Lock

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
CORS(app)  # Allow CORS for all domains

# -----------------------------
# Device check
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")


save_to_db_api_url  = "http://db_service:8000/users"

# -----------------------------
# Helper functions
# -----------------------------
def generate_hashed_uuid(length=8):
    full_uuid = str(uuid.uuid4())
    hash_object = hashlib.sha256(full_uuid.encode())
    hashed_uuid = hash_object.hexdigest()[:length]
    return hashed_uuid

def calculate_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)


# -----------------------------
# FaceRecognitionSystem class
# -----------------------------
class FaceRecognitionSystem:
    def __init__(self, folder_path='.'):
        # Config
        self.folder_path = folder_path

        # Models
        self.mtcnn = MTCNN(keep_all=False, device=device)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # State
        self.reference_embeddings = {}
        self.lock = Lock()

        # Threading states
        self.analysis_thread = None
        self.analysis_results = None
        
        # Face detection
        self.face_detected = False
        self.pending_input = False
        self.pending_face_data = None
        self.new_user_name = None

        # Load existing embeddings from JSON
        self.load_embeddings_from_folder()
        self.load_embeddings_from_db()

    def load_embeddings_from_folder(self):
        """Load face embeddings + metadata from the folder's JSON file."""
        self.reference_embeddings.clear()
        metadata_path = os.path.join(self.folder_path, "face_metadata.json")

        # Create file if missing
        if not os.path.exists(metadata_path):
            print("Metadata not found. Creating a new one...")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({"Customer": []}, f, indent=4, ensure_ascii=False)

        with open(metadata_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("Metadata file is corrupted. Reinitializing...")
                data = {"Customer": []}
                with open(metadata_path, "w", encoding="utf-8") as f2:
                    json.dump(data, f2, indent=4, ensure_ascii=False)

        if "Customer" in data:
            for customer in data["Customer"]:
                user_id = customer["id"]
                name = customer["name"]
                embedding = np.array(customer["embedding"])
                self.reference_embeddings[user_id] = (name, embedding)

        print(f"Loaded {len(self.reference_embeddings)} reference embeddings.")


    def load_embeddings_from_db(self):

        # print(f"save_to_db_api_url : {save_to_db_api_url}" , flush=True)
        # try : 
        #     response = requests.post(save_to_db_api_url, json=new_entry)

        #     if response.status_code == 201:
        #         print("User added successfully to DB!" , flush=True )
        #     else:
        #         print("Failed to add user to DB. Status code:", response.status_code , flush=True)
        # except Exception as e:
        #     print("Error saving to DB:", e , flush=True)



        """Load face embeddings + metadata from SQLite3 database."""
        self.reference_embeddings.clear()
        
        try:
            response = requests.get(save_to_db_api_url)
            if response.status_code == 200:
                data = response.json()
                for user in data:
                    user_id = str(user['u_id'])
                    name = user['name']
                    embedding = user['embedding']
                    self.reference_embeddings[user_id] = (name, embedding)

            print(f"Loaded {len(self.reference_embeddings)} reference embeddings from DB.")

        except sqlite3.Error as e:
            print(f"Database error: {str(e)}")


    def extract_embedding_and_boxes(self, image):
        """Extract face embedding and bounding boxes from given RGB image."""
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
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
            with torch.no_grad():
                embedding = self.facenet(face_tensor).cpu().numpy().flatten()
            return embedding, [boxes[0]], face
        else:
            return None, None, None

    def analyze_face_with_deepface(self, face_image):
        """Run DeepFace analysis (age, gender) on a face image."""
        try:
            analysis = DeepFace.analyze(face_image, actions=['age', 'gender'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            age = analysis.get('age', 'Unknown')
            # If gender is a dict of probabilities
            if isinstance(analysis.get('gender'), dict):
                gender_prob = analysis['gender']
                gender = max(gender_prob, key=gender_prob.get)
            else:
                gender = analysis.get('gender', 'Unknown')
            return age, gender
        except Exception as e:
            print(f"DeepFace error: {e}")
            return 'Unknown', 'Unknown'

    def start_analysis_thread(self, face_image):
        """Start a background thread to do the analysis if not already running."""
        if self.analysis_thread is None or not self.analysis_thread.is_alive():
            self.analysis_results = None
            self.analysis_thread = threading.Thread(target=self._analyze_face_in_background,
                                                    args=(face_image,))
            self.analysis_thread.start()

    def _analyze_face_in_background(self, face_image):
        """Internal function that runs in a separate thread."""
        results = self.analyze_face_with_deepface(face_image)
        with self.lock:
            self.analysis_results = results
        print("DeepFace analysis done in background:", results)

    def set_face_detected(self, value: bool):
        with self.lock:
            self.face_detected = value

    def get_face_detected(self) -> bool:
        with self.lock:
            return self.face_detected

    def save_new_face_and_embedding(self, embedding, face_image, user_name):
        """
        Save the new face embedding, along with age, gender (from analysis), to the JSON.
        """
        metadata_path = os.path.join(self.folder_path, "face_metadata.json")
        print(f"metadata_path : {metadata_path}",flush=True)

        # Load existing
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    print("Metadata file corrupted. Reinitializing...")
                    metadata = {"Customer": []}
        else:
            metadata = {"Customer": []}

        if "Customer" not in metadata:
            metadata["Customer"] = []
            print("customer not in metadata",flush=True)


        # We assume self.analysis_results is available after background thread
        with self.lock:
            print("while self LOCK",flush=True)

            if self.analysis_results is not None:
                age, gender = self.analysis_results
                print("analysis_result is not none",flush=True)
            else:
                age, gender = ('Unknown', 'Unknown')

        user_id = generate_hashed_uuid(4)
        print(f"Saving new user {user_name} with ID: {user_id}")
        new_entry = {
            "id": user_id,
            "name": user_name,
            "age": age,
            "gender": gender,
            "embedding": embedding.tolist()
        }
        metadata["Customer"].append(new_entry)

        # Save to JSON
        print("start saving json",flush=True)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        # Reload the local reference embeddings

        # self.load_embeddings_from_folder()        # Save to DB
        print(f"save_to_db_api_url : {save_to_db_api_url}" , flush=True)

        try : 
            response = requests.post(save_to_db_api_url, json=new_entry)

            if response.status_code == 201:
                print("User added successfully to DB!" , flush=True )
            else:
                print("Failed to add user to DB. Status code:", response.status_code , flush=True)
        except Exception as e:
            print("Error saving to DB:", e , flush=True)

        print(f"{user_name}'s embedding and analysis saved!", flush=True)

        print("load_embeddings_from_db",flush=True)
        # 데이터베이스에서 embedding 로드
        self.load_embeddings_from_db()


    def process_frame(self, frame):
        """
        Process a single frame:
          - Extract embedding
          - Compare to known embeddings
          - Handle new face logic
          - Return annotated frame, signals
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding, boxes, face_image = self.extract_embedding_and_boxes(frame_rgb)
        
        recognized_user = None
        new_face_detected = False

        if embedding is not None:
            self.set_face_detected(True)
            # Compare with known embeddings
            best_match = "No Match"
            min_dist = float('inf')

            for user_id, ref_data in self.reference_embeddings.items():
                ref_name, ref_embedding = ref_data
                dist = calculate_distance(ref_embedding, embedding)
                if dist < min_dist:
                    min_dist = dist
                    best_match = user_id if min_dist < 0.7 else "No Match"

            if best_match != "No Match":
                matched_name = self.reference_embeddings[best_match][0]
                recognized_user = matched_name

                # Draw bounding box
                if boxes:
                    x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        matched_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
            else:
                # Start analysis if needed
                self.start_analysis_thread(face_image)

                # Draw bounding box for No Match
                if boxes:
                    x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "No Match",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2
                    )
                new_face_detected = True
                self.pending_face_data = (embedding, boxes, face_image)
        else:
            # No face
            self.set_face_detected(False)

        return frame, recognized_user, new_face_detected

# -----------------------------
# Create a single system instance
# -----------------------------
face_system = FaceRecognitionSystem(folder_path='.')

# -----------------------------
# Routes
# -----------------------------
@app.route('/video')
def video():
    return Response(stream_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def stream_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Failed to open webcam.")
        return  # Or yield nothing

    # simple timing for recognized user
    recognized_user = None
    recognized_time = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        processed_frame, user_name, new_face = face_system.process_frame(frame)
        
        # If recognized
        if user_name is not None:
            if recognized_user is None:
                recognized_user = user_name
                recognized_time = time.time()
                print(f"Welcome, {recognized_user}. 3 seconds to exit...")

            # After 3 seconds, break
            if recognized_user == user_name and (time.time() - recognized_time) > 3:
                print(f"Goodbye, {recognized_user}. Exiting stream...")
                break

        else:
            # reset recognized user if no match
            recognized_user = None

        # If new face is detected, could do additional logic or let user input name from the front-end
        # This is the place to set face_system.pending_input = True if you want to enforce it in code
        # But we'll rely on the /submit_name route for that.

        ret2, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret2:
            print("Failed to encode frame.")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

    cap.release()

@app.route('/submit_name', methods=['POST'])
def submit_name():
    # We assume the front-end calls this after user typed a new name
    data = request.json
    new_name = data.get('name', None)
    if not new_name:
        return jsonify({"error": "Name is required"}), 400

    # Check if there's pending data
    if not face_system.pending_face_data:
        return jsonify({"error": "No face pending for input"}), 400

    embedding, boxes, face_image = face_system.pending_face_data
    with face_system.lock:
        results = face_system.analysis_results  # we want to ensure deepface analysis done

    if results is None:
        return jsonify({"error": "Analysis results not ready"}), 400

    # Save the new face
    face_system.save_new_face_and_embedding(embedding, face_image, new_name)

    # Clear pending data
    face_system.pending_face_data = None
    face_system.analysis_results = None
    return jsonify({"message": f"Face saved for {new_name}"}), 200

@app.route('/check_face', methods=['GET'])
def check_face():
    return jsonify({"face_detected": face_system.get_face_detected()})

if __name__ == '__main__':
    # Note: threaded=True can help with concurrency on dev server, but not recommended for production
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
