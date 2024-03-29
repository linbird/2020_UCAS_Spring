from multiprocessing import Process, Pool
import os
import cv2
import numpy as np
import face_detect.face_location as location
import face_recog.face_recogn as face_recognition
from multiprocessing import Pool
from multiprocessing import cpu_count

import time

import onnx
# import vision.utils.box_utils_numpy as box_utils
import face_detect.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort



label_path = "/home/linbird/2020_UCAS_Spring/Face_Identify/fusion/face_detect/models/voc-model-labels.txt"

onnx_path = "/home/linbird/2020_UCAS_Spring/Face_Identify/fusion/face_detect/models/onnx/version-RFB-320.onnx"
# class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
# onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture("/home/linbird/video.mp4")

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file(
    "/home/linbird/下载/obama.jpeg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
]
known_face_names = [
    "BO",
]

while True:
    # Grab a single frame of video
    time_start = time.time()
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:,:,::-1]
    
    # time_time = time.time()
    # Find all the faces and face enqcodings in the frame of video
    face_locations = location.detect(rgb_frame, ort_session, input_name)
    # face_locations = face_recognition.face_locations(rgb_frame)
    # print("face_locations cost time:{}".format(time.time() - time_time))
    # face_num = len(face_locations)

    # time_time = time.time()
    face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations)

    # print("face_encodings cost time:{}".format(time.time() - time_time))
    # print(face_encodings)
    # time_time = time.time()

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        if name == "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    
    fps = "FPS:" + str(format(1/(time.time()-time_start), '.2f'))
    cv2.putText(frame, fps, (5,15),cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
    # Hit 'q' on the keyboard to quit!
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
