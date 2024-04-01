import face_recognition
import pickle
import cv2
import copy
import numpy as np
from pathlib import Path
from collections import Counter
from imageTrainTest import encode_known_faces

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)


# allows for faces within an image to be identified 
def recognize_faces(
    input_image,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    display: bool = True,
) -> None:
    with encodings_location.open(mode="rb") as f: # load in the known encodings 
        loaded_encodings = pickle.load(f)

    #input_image = face_recognition.load_image_file(image_location) # loads in the image to be analysed 

    # detect faces in the image and get their encodings
    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    output_image = copy.deepcopy(input_image)
    names_list = []

    # looped so multiple faces within the image can be detected
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name: # unkown faces are marked accordingly 
            name = "Unknown"
            names_list.append('stranger')
        else: 
            names_list.append(name)

        if display:
            #_display_face(draw, bounding_box, name) # used to generate display with labelled bounding boxes over faces 
            output_image = _display_face(output_image, bounding_box, name) # used to generate display with labelled bounding boxes over faces

    if display:
        # Display the resulting image
        cv2.imshow('Video', output_image)

    return names_list


# helper function to take the new and load encoding and match the encodings 
# where possible to return the matching encoding where possible and None where not 
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    

# displays an image with the detected faces in bounding boxes with their respective labels
def _display_face(output_image, bounding_box, name):
    
    top, right, bottom, left = bounding_box
    # Draw a box around the face
    output_image = cv2.rectangle(output_image, (left, top), (right, bottom), (255, 0, 0), 2)

    (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Prints the text.
    output_image = cv2.rectangle(output_image, (left, bottom + h), (left + w, bottom), (255, 0, 0), -1)
    output_image = cv2.putText(output_image, name, (left, bottom + h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
    return output_image


# # reads in a video input and performs facial recognition on it
# def video(model: str = "hog", 
#     display: bool = True):

#     cap= cv2.VideoCapture("MiroCameraOutput.avi") # Video file input
#     #cap= cv2.VideoCapture(0) # Webcam input

#     while True:

#         # Grab a single frame of video
#         ret, frame = cap.read()

#         if frame is None:
#             break

#         # Convert the image from BGR colour, used by OpenCV, to RGB colour, used by face_recognition
#         rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

#         # perform facial recognition
#         names = recognize_faces(rgb_frame, 
#                                 model=model,
#                                 display=display)
#         if all(names) == True: # checks if all detected faces are unknown  
#             print("ALARM!")

#         # Wait for Enter key to stop
#         if cv2.waitKey(25) == 13:
#             break
    

# #encode_known_faces() # run to train model on faces within the trianing folder 
# video()