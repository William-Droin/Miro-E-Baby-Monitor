from flask import Flask, Response, render_template, jsonify
import threading
import videoDetector
import cv2
import numpy as np
import time
from threading import Lock

# Create a lock
people_in_room_lock = Lock()


ros_enable = False
people_in_room = ['initialising face detection']

if ros_enable == True:
    import rospy
    from video_stream import ObjectDetector
    object_detector = ObjectDetector()
else:
    camera = cv2.VideoCapture(0)

app = Flask(__name__)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Variable to keep track of face recognition status
face_recognized = False

# Name of the people on screen


# Callback function to be called when a face is recognized
def on_face_recognized():
    global face_recognized
    if not face_recognized:
        face_recognized = True
        move_stop()
        send_notif("Face Detected", "A face has been detected. The robot is stopped.")
        # Additional logic to handle the event (e.g., stop video stream, update web app) can be added here

# Initialize face recognition in a separate thread to not block the Flask server
def initialise_face_recognition():
    global people_in_room

    print("Thread is running")
    while True:
        try:
            time.sleep(5)
            cam = cv2.VideoCapture('http://127.0.0.1:5001/video_feed')
            while not cam:
                print('waiting for stream to start')
                cam = cv2.VideoCapture('http://127.0.0.1:5001/video_feed')
                time.sleep(2)
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break  # Exit loop if we can't grab a frame

            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

            people_in_room_temp = videoDetector.recognize_faces(rgb_frame, model="hog", display=False)

            # Safely update the global variable
            with people_in_room_lock:
                people_in_room = people_in_room_temp
        except Exception as e:
            print(f"Error in face recognition thread: {e}")
            break

        # This delay is crucial to not overload the CPU
        time.sleep(0.1)

def get_latest_notification():
    # Example: Returning the current time as a notification
    with people_in_room_lock:
        listpeople = people_in_room.copy()
    if listpeople == []:
        return {"message": "I don't see anybody near the baby"}
    elif list(set(listpeople))[0] == 'stranger':
        return {"message": "Only strangers are near the baby"}
    else:
        return {"message": f"I can see that {', '.join(listpeople)}, are near the baby"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-notification')
def get_notification():
    # Simulate checking for a new notification
    notification = get_latest_notification()
    return jsonify(notification)

@app.route('/video_feed')
def video_feed():
    if ros_enable == True:
        return Response(object_detector.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Control buttons routes
@app.route('/forward', methods=['POST'])
def move_forward():
    print("Moving forward")
    return jsonify(success=True)

@app.route('/backward', methods=['POST'])
def move_backward():
    print("Moving backward")
    return jsonify(success=True)

@app.route('/left', methods=['POST'])
def move_left():
    print("Moving left")
    return jsonify(success=True)

@app.route('/right', methods=['POST'])
def move_right():
    print("Moving right")
    return jsonify(success=True)


def move_stop():
    print("Stopped")

def send_notif(type, text):
    print(f"{type}: {text}")

if __name__ == '__main__':
    #rospy.init_node('object_detector', anonymous=True)
    # face_recognition_thread = threading.Thread(target=initialise_face_recognition)
    # face_recognition_thread.start()
    

    threading.Thread(target=lambda: app.run(debug=False, use_reloader=False, port=5001, host='0.0.0.0')).start()
    initialise_face_recognition()
