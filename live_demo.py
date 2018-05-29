from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
# from utils.inference import detect_faces
# from utils.inference import draw_text
# from utils.inference import draw_bounding_box
# from utils.inference import apply_offsets
# # from utils.inference import load_detection_model
# # from utils.preprocessor import preprocess_input
from inference import *
from preprocessor import *

# parameters for loading data and images
# detection_model_path = '/Users/gowthamkannan/CV_PROJ/Classifier.h5'
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'Classifier_10.h5'
# emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
# emotion_labels = get_labels('fer2013')
emotion_labels = sorted(['NONE','UNCERTAIN','SAD','NEUTRAL','DISGUST','ANGER','SURPRISE','FEAR','CONTEMPT','HAPPY'])

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
# print(emotion_target_size)
# emotion_target_size= list(emotion_target_size)
# emotion_target_size.insert(2, 3)
# emotion_target_size=tuple(emotion_target_size)

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        print("Check")    
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (300,300))
            # print(gray_face.shape)
        except Exception as e:
            print(e)
            continue

        gray_face = preprocess_input(gray_face, True)
        # gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        gray_face=np.resize(gray_face,(1,300,300,3))
        # print(gray_face_temp.resize((224,224,3)))
        # print(type(gray_face),gray_face.shape)
        # print(gray_face.shape)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = emotion_prediction.argmax()
        emotion_text = emotion_labels[emotion_label_arg]
        print(emotion_text,emotion_probability)
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        
        if  'ANGER' in emotion_text:
            color=emotion_probability*np.asarray((255,0,0))
        elif 'CONTEMPT' in emotion_text:
            color=emotion_probability*np.asarray((255,128,0))
        elif 'DISGUST' in emotion_text:
            color=emotion_probability*np.asarray((255,102,102))
        elif 'FEAR' in emotion_text:
            color=emotion_probability*np.asarray((0,0,0))
        elif 'HAPPY' in emotion_text:
            color=emotion_probability*np.asarray((51,102,0))
        elif 'NEUTRAL' in emotion_text:
            color=emotion_probability*np.asarray((255,255,255))
        elif 'NONE' in emotion_text:
            color=emotion_probability*np.asarray((51,0,0))
        elif 'SAD' in emotion_text:
            color=emotion_probability*np.asarray((0,255,255))
        elif 'SURPRISE' in emotion_text:
            color=emotion_probability*np.asarray((255,255,0))
        elif 'UNCERTAIN' in emotion_text:
            color=emotion_probability*np.asarray((0,0,255))



        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
