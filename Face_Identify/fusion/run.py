from multiprocessing import Process, Pool
import sys
import os
import cv2
from colorama import Fore
from colorama import Style
import numpy as np
import face_detect.face_location as location
import face_recog.face_recogn as face_recognition
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
import threading
from queue import Queue

import time

import onnx
# import vision.utils.box_utils_numpy as box_utils
import face_detect.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort

def mp_encoding(face_image, face_locations):
    res =Queue()
    pths = []
    results = []
    for face_location in face_locations:
        pths.append(threading.Thread(target=face_recognition.mp_face_encodings, args=(rgb_frame, res, [face_location])))
        pths[-1].start()
    for pth in pths:
        pth.join()
    for _ in range(len(face_locations)):
        results.append((res.get()[0]))
    return results
    # print(results)


cwd = os.getcwd()
label_path = cwd + "/data/models/voc-model-labels.txt"
onnx_path = cwd + "/data/models/onnx/version-RFB-320.onnx"
# class_names = [name.strip() for name in open(label_path).readlines()]
predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)

# onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

known_face_encodings = []
known_face_names = []

# Load a sample picture and learn how to recognize it.
for person in os.listdir(cwd + "/data/photo/"):
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(cwd + "/data/photo/" + person))[0])
    #img = face_recognition.load_image_file(cwd + "/data/photo/" + person)
    #face_encoding = face_recognition.face_encodings(img)[0]
    #print(face_encoding)
    #known_face_encodings.append(face_encoding)
    known_face_names.append(os.path.splitext(person)[0])
    print(Fore.GREEN + 'OK: ' + Style.RESET_ALL + 'load ' + os.path.splitext(person)[0])

if(sys.argv[1] != "online"):
    video_capture = cv2.VideoCapture(os.getcwd() + "/data/video.mp4")
else:
    video_capture = cv2.VideoCapture(0)
# Create arrays of known face encodings and their names

while True:
    time_start = time.time();
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print(Fore.RED + "ERROR:" + Style.RESET_ALL + "no frame input" )
        break
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:,:,::-1]

    time1 = time.time()
    # Find all the faces and face enqcodings in the frame of video
    face_locations = location.detect(rgb_frame, ort_session, input_name)
    time2 = time.time()
#多进程入不敷出
##    print(Fore.GREEN + "创建多进程： " + Style.RESET_ALL)
#    if len(face_locations) <= 1:
#       face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#       locations_encoding = zip(face_locations, face_encodings)
#    else:
#        pool = Pool(min(cpu_count()-1, len(face_locations)))
#        locations_encoding = []
#        for face_location in face_locations:
#            #print("执行进程")
#            #face_encoding = pool.apply_async(face_recognition.face_encodings, (rgb_frame, [face_location])).get()
#            #print(face_encoding)
#            face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])
#            #face_encodings.append(face_encoding[0])
#            locations_encoding.append([face_location, face_encoding[0]])
##       face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#        pool.close()
#        pool.join()
##    
#    
##    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#    time4 = time.time()
##    print(time2 - time1, time4 - time2)
 
    # face_encodings = mp_encoding(rgb_frame, face_locations)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    # print(face_encodings)
#    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Loop through each face in this frame of video
    #for (top, right, bottom, left), face_encoding in locations_encoding:
        # See if the face is a match for the known face(s)
        # print(face_encoding)
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

    # print(time.time() - time4)
            # Hit 'q' on the keyboard to quit!
    fps = "FPS:" +  str(format((1/(time.time() - time_start)), '0.2f'))
    cv2.putText(frame, fps, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,255), 1)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
