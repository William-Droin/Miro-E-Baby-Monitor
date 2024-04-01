from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO
import threading
from datetime import datetime

ros_enable = False

if ros_enable == True:
    import rospy
    from video_stream import ObjectDetector
    object_detector = ObjectDetector()
else:
    import cv2
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
    print("Face recognition initialized")
    # Assuming your ObjectDetector has a method to start face recognition and accept a callback
    #object_detector.start_face_recognition(callback=on_face_recognized)

def get_latest_notification():
    # Example: Returning the current time as a notification
    return {"message": f"Notification at {datetime.now().strftime('%H:%M:%S')}"}

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
    face_recognition_thread = threading.Thread(target=initialise_face_recognition)
    face_recognition_thread.start()
    app.run(app, host='0.0.0.0', port=5000, debug=True)
