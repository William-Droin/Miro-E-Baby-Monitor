import rospy
from flask import Flask, Response
from video_stream import ObjectDetector

app = Flask(__name__)
object_detector = ObjectDetector()

@app.route('/')
def video_feed():
    return Response(object_detector.generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

 

if __name__ == '__main__':
    rospy.init_node('object_detector', anonymous=True)
    app.run(host='0.0.0.0', port=3000, debug=True, threaded=True)