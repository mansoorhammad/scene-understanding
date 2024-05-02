import time
from cv2 import imshow, imwrite
import cv2
from flask import Flask, flash, render_template, Response #redirect, url_for,
from ultralytics import YOLO
from transformers import pipeline
from cv2 import *
import math
import random

app = Flask(__name__)

print("Caption generation - inside the : =============START")
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

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
print("Caption generation - inside the : =================END ")

def generate_framesIC(camera):
    print("Caption generation - inside the : generate_framesIC ")
  #  while True:
        # last_incremented_time = time.time()  # Record the start time
    print("Caption generation - inside the : while loop Priyanka")
    success, frame = camera.read()

    # Convert the frame to JPEG
   # frame = cv2.imencode('.jpg', frame)[1]
    #print(frame)

    # Save the frame as a JPEG image
    imshow("GeeksForGeeks", frame) 
  
    # saving image in local storage 
    imwrite("frame.png", frame) 
  
    # Release the camera
    #camera.release()
  
    #if not success:
     #   print("Caption Generation - Reading camera error")
      #  break
    #else:
    loadText = "Model is generating the text"
    text=image_to_text("frame.png")
    text=' '.join(str(x) for x in text)
    substring = text[19:]
    substring=substring.rstrip("}]")
    #print(text)
    return substring

        
# def videoIC(camera):
#     for text in generate_framesIC(camera):
#         return Response(text, mimetype='text/plain')
