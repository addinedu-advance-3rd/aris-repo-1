import cv2
from deepface import DeepFace

import tensorflow as tf

# GPU 장치 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 메모리 증가 허용
        print(f"Available GPUs: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print("Error configuring GPU:", e)
else:
    print("No GPUs available")


# List of available backends, models, and distance metrics
backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]



def realtime_face_recognition():
    # Define a video capture object
    vid = cv2.VideoCapture(0)

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()

        # Perform face recognition on the captured frame
        # Find faces and identify people using a specific model and distance metric
        people = DeepFace.find(img_path=frame, db_path="./", model_name=models[4], distance_metric=metrics[2], enforce_detection=False)

        for person in people:
            # Retrieve the coordinates of the face bounding box
            x = person['source_x'][0]
            y = person['source_y'][0]
            w = person['source_w'][0]
            h = person['source_h'][0]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Get the person's name and display it on the image
            name = person['identity'][0].split('/')[1]
            cv2.putText(frame, name, (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 960, 720)
        cv2.imshow('frame', frame)

        # Check if the 'q' button is pressed to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    vid.release()
    cv2.destroyAllWindows()

# Perform real-time face recognition using the webcam
realtime_face_recognition()