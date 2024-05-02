import random
from ultralytics import YOLO
import cv2
import math
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

#Generating a dictionary of class names with their respective colors
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# List of commonly used colors
common_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                 (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                 (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
                 (255, 128, 128), (128, 255, 128), (128, 128, 255), (192, 192, 192), (128, 128, 128),
                 (255, 255, 255), (0, 0, 0), (255, 255, 128), (255, 128, 255), (128, 255, 255), (192, 0, 0),
                 (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (255, 192, 0),
                 (255, 0, 192), (192, 255, 0), (0, 255, 192), (192, 0, 255), (0, 192, 255), (255, 192, 192),
                 (192, 255, 192), (192, 192, 255), (224, 224, 224), (160, 160, 160), (128, 64, 0), (0, 128, 64),
                 (64, 0, 128), (128, 128, 64), (128, 64, 128), (64, 128, 128), (192, 128, 64), (192, 64, 128),
                 (128, 192, 64), (64, 192, 128), (128, 64, 192), (64, 128, 192), (192, 128, 192), (192, 192, 128),
                 (128, 192, 192), (255, 224, 0), (255, 0, 224), (224, 255, 0), (0, 255, 224), (224, 0, 255),
                 (0, 224, 255), (255, 224, 224), (224, 255, 224), (224, 224, 255), (255, 255, 192), (255, 192, 255),
                 (192, 255, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255), (255, 192, 128), (255, 128, 192),
                 (192, 255, 128), (128, 255, 192), (192, 128, 255), (128, 192, 255), (224, 192, 128), (224, 128, 192),
                 (192, 224, 128), (128, 224, 192), (192, 128, 224), (128, 192, 224), (224, 192, 192), (192, 224, 192),
                 (192, 192, 224)]
dark_common_colors = [
    (0, 0, 0),         # Black
    (25, 25, 112),     # MidnightBlue
    (47, 79, 79),      # DarkSlateGray
    (0, 100, 0),       # DarkGreen
    (139, 0, 0),       # DarkRed
    (139, 69, 19),     # SaddleBrown
    (139, 0, 139),     # DarkMagenta
    (128, 128, 0),     # Olive
    (0, 128, 128),     # Teal
    (139, 69, 19),     # SaddleBrown
    (72, 61, 139),     # DarkSlateBlue
    (205, 92, 92),     # IndianRed
    (128, 0, 0),       # Maroon
    (85, 107, 47),     # DarkOliveGreen
    (0, 0, 139),       # DarkBlue
    (47, 79, 79),      # DarkSlateGray
    (0, 139, 139),     # DarkCyan
    (0, 128, 0),       # DarkGreen
    (0, 0, 139),       # DarkBlue
    (0, 139, 0),       # DarkGreen
    (139, 0, 0),       # DarkRed
    (128, 0, 128),     # DarkMagenta
    (139, 69, 19),     # SaddleBrown
    (139, 0, 139),     # DarkMagenta
    (128, 128, 0),     # Olive
    (47, 79, 79),      # DarkSlateGray
    (139, 0, 0),       # DarkRed
    (139, 69, 19),     # SaddleBrown
    (128, 128, 0),     # Olive
    (0, 0, 139),       # DarkBlue
    (0, 128, 128)      # Teal
]

# Generate a dictionary with random common RGB values for each class name
class_colors = {}
random.shuffle(dark_common_colors)

for i, class_name in enumerate(classNames):
    color = dark_common_colors[i % len(dark_common_colors)]
    class_colors[class_name] = color

######### Utilizing YOLO-seg Model for Webcam Data and Displaying Results ######

def generate_framesSemanticSegmentation(camera):
    # cap=cv2.VideoCapture(0)

    frame_width=int(camera.get(3))
    frame_height = int(camera.get(4))
    


    out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    model=YOLO("../YOLO-Weights/yolov8n-seg.pt")
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
                ]
    while True:
        success, img = camera.read()

        results=model(img,stream=False,verbose=False)
        for result in results:
            frame = result.plot(boxes=True,labels=True,masks=True,probs=True)

            # Display the annotated frame
            # cv2.imshow("YOLOv8 Inference", frame)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            
        # if cv2.waitKey(1) & 0xFF==ord('q'):
        #     break
    out.release()

def videoSemanticSegmentation(camera):
    return Response(generate_framesSemanticSegmentation(camera),mimetype='multipart/x-mixed-replace; boundary=frame')
