# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import threading
import os

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def gen_bound_color(self, boxes):
        red_bound = set()
        blue_bound = set()
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                tot = 0
                for num in range(4):
                    tot += abs(boxes[i][num]-boxes[j][num])
                if tot < 200:
                    red_bound.add(boxes[i])
                    red_bound.add(boxes[j])
                else:
                    blue_bound.add(boxes[i])
        blue_bound = blue_bound - red_bound
        return blue_bound, red_bound

    def close(self):
        self.sess.close()
        self.default_graph.close()

def announce(item):
    if (0 <= item[1] < 668) and (0 <= item[0] < 225):
        os.system('say "Please maintain social distancing near the Gap store."')
    elif (0 <= item[1] < 668) and (225 <= item[0] < 720):
        os.system('say "Please maintain social distancing near the South sidewalk."')
    elif (668 <= item[1] <= 1280) and (0 <= item[0] < 225):
        os.system('say "Please maintain social distancing near the North sidewalk."')
    else:
        os.system('say "Please maintain social distancing near the newspaper stand"')

if __name__ == "__main__":
    # model_path = './model_slow/frozen_inference_graph.pb'
    # model_path = './model_fast/frozen_inference_graph.pb'
    model_path = './model_v_fast/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture('./video.avi')
    th = threading.Thread()
    while True:
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.
        check_box = []
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                check_box.append(boxes[i][:])

        blue_bound, red_bound = odapi.gen_bound_color(check_box)
        for item in blue_bound:
            cv2.rectangle(img,(item[1],item[0]),(item[3],item[2]),(255,0,0),2)
        for item in red_bound:
            cv2.rectangle(img,(item[1],item[0]),(item[3],item[2]),(0,0,255),2)
            if not th.isAlive():
                th = threading.Thread(name='announce', target=announce, args=[(item[0], item[1])])
                th.start()

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
