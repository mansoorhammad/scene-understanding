from flask import Flask, render_template, Response #redirect, url_for,
from ultralytics import YOLO
import cv2
import math
import random

app = Flask(__name__)


# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
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
class_colors = {}
random.shuffle(dark_common_colors)

for i, class_name in enumerate(classNames):
    color = dark_common_colors[i % len(dark_common_colors)]
    class_colors[class_name] = color

def generate_framesOD(camera):
    while True:
        success, frame = camera.read()
        results = model(frame, stream=True, verbose=False)

        # coordinates
        
        
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}'
                a, b, c = class_colors[class_name]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (a,b,c), 3)
                t_size = cv2.getTextSize(label, 4, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1,y1), c2, [a,b,c], -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, label, (x1,y1-2),4, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        
            if not success:
                break
            else:
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def videoOD(camera):
    return Response(generate_framesOD(camera),mimetype='multipart/x-mixed-replace; boundary=frame')
