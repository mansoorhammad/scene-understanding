import random
import time
from flask import Flask, jsonify, render_template, Response, request #redirect, url_for,
from ultralytics import YOLO
import cv2
import math
from Image_captioning import generate_framesIC
from object_detection import videoOD
from semantic_segmentation import videoSemanticSegmentation
import atexit

app = Flask(__name__)

camera=cv2.VideoCapture(0)

# camera.set(3, 240)
# camera.set(4, 480)

width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("Current camera width:", width)
print("Current camera height:", height)

# aspect_ratio = width / height

# Define the maximum height that fits within your grid space
# max_display_height = 480  # Example maximum height

# Calculate the new height based on the maximum height and original aspect ratio
# new_height = min(height, max_display_height)
# new_width = int(new_height * aspect_ratio)

# # Resize the frame
# resized_frame = cv2.resize(camera, (new_width, new_height))


# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# widthnew = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
# heightnew = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

# print("Current camera width new :", widthnew)
# print("Current camera height new :", heightnew)

caption_text = "loading"

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


#loading the html
@app.route('/')
def index():
    return render_template('liveStreaming.html', caption_text=caption_text)
    
#input video is streaming in the top left of the screen SCREEN 1
@app.route('/video')
def video():
    try:
        return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"An error occured input video SCREEN 1 : {str(e)}")
        return "An unexpected error occurred.", 500

#object detection SCREEN 2 
@app.route('/video_objectDetection')
def video_objectDetection():
    try:
        return videoOD(camera)
    
    except Exception as e:
        print(f"An error occured Object detection SCREEN 2 : {str(e)}")
        return "An unexpected error occurred.", 500

#semantic segmentation SCREEN 3
@app.route('/video_semanticSegmentation')
def video_semanticSegmentation():
    try:
        return videoSemanticSegmentation(camera)
    except Exception as e:
        print(f"An error occured sematic segmentation SCREEN 3 : {str(e)}")
        return "An unexpected error occurred.", 500

@app.route('/video_captionGeneration', methods=["POST"])
def video_captionGeneration():
    global caption_text
    caption_text = generate_framesIC(camera)
    print(caption_text)
    return render_template('liveStreaming.html', caption_text=caption_text)
 

def release_camera():
    print("Release camera resources...")
    camera.release()


cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)