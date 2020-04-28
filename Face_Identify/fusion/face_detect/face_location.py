import time

import cv2
import numpy as np
import onnx
# import vision.utils.box_utils_numpy as box_utils
import face_detect.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort


# label_path = "/home/linbird/2020_UCAS_Spring/Face_Identify/fusion/face_detect/models/voc-model-labels.txt"

# onnx_path = "/home/linbird/2020_UCAS_Spring/Face_Identify/fusion/face_detect/models/onnx/version-RFB-320.onnx"
# # class_names = [name.strip() for name in open(label_path).readlines()]

# predictor = onnx.load(onnx_path)
# onnx.checker.check_model(predictor)
# # onnx.helper.printable_graph(predictor.graph)
# predictor = backend.prepare(predictor, device="CPU")  # default CPU

# ort_session = ort.InferenceSession(onnx_path)
# input_name = ort_session.get_inputs()[0].name


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    print("starting predict")
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def detect(orig_image, ort_session, input_name):
    print("starting detect")
    threshold = 0.7
    if orig_image is None:
        print("no img")
        # break
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    # confidences, boxes = predictor.run(image)
    print("starting ort_session.run")
    confidences, boxes = ort_session.run(None, {input_name: image})
    print("end ort_session.run")

    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    # print([(box[1], box[2], box[3], box[0]) for box in boxes])
    return [(box[1], box[2], box[3], box[0]) for box in boxes]
    #(top, right, bottom, left)
