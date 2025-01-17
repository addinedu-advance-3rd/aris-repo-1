from deepface import DeepFace

DeepFace.stream("/home/addinedu/Downloads/images", enable_face_analysis=True, anti_spoofing=True, time_threshold = 1, 
)
                # model_name="DeepFace")