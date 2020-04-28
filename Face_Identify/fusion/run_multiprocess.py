# import face_recognition
import face_recog.face_recogn as face_recognition

import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy as np
import threading
import platform
import face_detect.face_location as location

# This is a little bit complicated (but fast) example of running face recognition on live video from your webcam.
# This example is using multiprocess.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

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

# options = ort.SessionOptions()
# options.enable_sequential_execution = False
# options.session_thread_pool_size = 3

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# Get next worker's id
def next_id(current_id, worker_num):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1


# Get previous worker's id
def prev_id(current_id, worker_num):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1


# A subprocess use to capture frames.
def capture(read_frame_list, Global, worker_num):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    # video_capture.set(3, 640)  # Width of the frames in the video stream.
    # video_capture.set(4, 480)  # Height of the frames in the video stream.
    # video_capture.set(5, 30) # Frame rate.
    print("Width: %d, Height: %d, FPS: %d" %
          (video_capture.get(3), video_capture.get(4), video_capture.get(5)))

    while not Global.is_exit:
        # If it's time to read a frame
        if Global.buff_num != next_id(Global.read_num, worker_num):
            # Grab a single frame of video
            ret, frame = video_capture.read()
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num, worker_num)
            # print("reading")
            # cv2.imshow('Video', frame)
        # else:
        #     # print("sleep")
        #     time.sleep(0.01)

    # Release webcam
    video_capture.release()

##将人脸编码任务均分到每个线程
def encode_multi_process(worker_id, rgb_frame, face_locations, Global):
    for worker_id in range(1, Global.worker_num + 1):
        p.append(Process(target=encode_multi_process, args=(
            worker_id, read_frame_list, write_frame_list, Global, worker_num,)))
        p[worker_id].start()
        # Wait to write
    while Global.write_num != worker_id:
        time.sleep(0.01)
    # Send frame to global
    write_frame_list[worker_id] = frame_process
    # Expect next worker to write frame
    Global.write_num = next_id(Global.write_num, worker_num)
    face_recognition.face_encodings(rgb_frame, face_locations)


# Many subprocess use to process frames.
def process(read_frame_list, write_frame_list, Global):
    print("process")
    known_face_encodings = Global.known_face_encodings
    known_face_names = Global.known_face_names
    while not Global.is_exit:
        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            # If the user has requested to end the app, then stop waiting for webcam frames
            if Global.is_exit:
                break

            time.sleep(0.01)

        # Delay to make the video look smoother
        # time.sleep(Global.frame_delay)

        # Read a single frame from frame list
        frame_process = read_frame_list[worker_id]

        # Expect next worker to read frame
        Global.read_num = next_id(Global.read_num, worker_num)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame_process[:, :, ::-1]
        face_locations = location.detect(rgb_frame, ort_session, input_name)

        # time_time = time.time()
        # print("encodings time:{}".format(time.time() - time_time))

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face
            cv2.rectangle(frame_process, (left, top),
                          (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame_process, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_process, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)



def load_know_person():
    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file(
        "/home/linbird/2020_UCAS_Spring/Face_Identify/face_recognition/examples/registered/obama.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file(
        "/home/linbird/2020_UCAS_Spring/Face_Identify/face_recognition/examples/registered/biden.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Create arrays of known face encodings and their names
    Global.known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding
    ]
    Global.known_face_names = [
        "Barack Obama",
        "Joe Biden"
    ]

def presetting():
    # Fix Bug on MacOS
    if platform.system() == 'Darwin':
        set_start_method('forkserver')

    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    # Number of workers (subprocess use to process frames)
    if cpu_count() > 2:
        Global.worker_num = cpu_count() - 1  # 1 for capturing frames
    else:
        Global.worker_num = 2


if __name__ == '__main__':
    presetting()
    # Subprocess list
    p = []

    load_know_person()

    # Create a thread to capture frames (if uses subprocess, it will crash on Mac)
    p.append(threading.Thread(target=capture, args=(
        read_frame_list, Global, worker_num,)))
    p[0].start()

    # Start to show video
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)
            # Display the resulting image
            cv2.imshow('Video', write_frame_list[prev_id(
                Global.write_num, worker_num)])

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Global.is_exit = True
            break

        time.sleep(0.01)

    # Quit
    cv2.destroyAllWindows()
