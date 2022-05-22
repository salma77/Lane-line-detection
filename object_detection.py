# Importing libraries
import os
import cv2
import numpy as np
from lane_lines import lane_finding_pipeline

cfg_path = os.path.join("yolo", "yolov3.cfg")
weights_path = os.path.join("yolo", "yolov3.weights")
names_path = os.path.join("yolo", "coco.names")
test_image_path = os.path.join("test_images", "test1.jpg")
nms_thr = 0.5
confidence_thr = 0.6

with open(names_path, "r") as fp:
    names = fp.read().strip().split("\n")

yolo = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

def get_output_layers(net):
    """
    Get the names of the output layers of a network.

    Args:
        net: The network to get the output layers from.

    Returns:
        A list of the names of the output layers.
    """
    return list(net.getUnconnectedOutLayersNames())


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draw a bounding box.

    Args:
        img: The image to draw on.
        class_id: The class id of the object detected.
        confidence: The confidence of the object detected.
        x: The x coordinate of the top left corner of the bounding box.
        y: The y coordinate of the top left corner of the bounding box.
        x_plus_w: The x coordinate of the bottom right corner of the bounding box.
        y_plus_h: The y coordinate of the bottom right corner of the bounding box.

    Returns:
        The image with the bounding box drawn.
    """
    label = str(names[class_id])

    color = (0, 255, 255)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_bounding_boxes(outs, W, H):
    """
    Get the bounding boxes from the output layer.

    Args:
        outs: The output layer of the network.
        W: The width of the image.
        H: The height of the image.

    Returns:
        A tuple of the class ids, confidences, and the bounding boxes.
    """
    # initialization
    class_ids = []
    confidences = []
    boxes = []

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_thr:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    return class_ids, confidences, boxes


def nms(img, class_ids, confidences, boxes):
    """
    Perform non-maximum suppression.
    
    Args:
        img: The image to draw on.
        class_ids: The class ids of the objects detected.
        confidences: The confidences of the objects detected.
        boxes: The bounding boxes of the objects detected.
    """
    global confidence_thr, nms_thr
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thr, nms_thr)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(img, class_ids[i], confidences[i], round(
            x), round(y), round(x+w), round(y+h))


def detection_pipeline(img):
    """
    Perform the object detection pipeline.
    
    Args:
        img: The image to process.
    
    Returns:
        The image with the objects detected.
    """
    global yolo
    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255., (416, 416),
                                 crop=False, swapRB=False)
    yolo.setInput(blob)
    outs = yolo.forward(get_output_layers(yolo))
    nms(img, *get_bounding_boxes(outs, W, H))
    return img


def combined_pipeline(img):
    """
    Perform the combined pipeline of lane detection and object detection.
    
    Args:
        img: The image to process.
    
    Returns:
        The image with the bounding boxes drawn and lane lines detected.
    """
    img = lane_finding_pipeline(img)
    return detection_pipeline(img)
