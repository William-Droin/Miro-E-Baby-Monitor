import face_recognition
import pickle
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


# trains the model to recognize the desired peoples' faces
def encode_known_faces(
    # utilising histogram of oriented gradients for object detection 
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH 
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"): # learns every person defined in the training folder
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model) # detects the locations of faces
        face_encodings = face_recognition.face_encodings(image, face_locations) # generates encodings for the detected faces 

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


# allows for faces within an image to be identified 
def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    display: bool = True,
) -> None:
    with encodings_location.open(mode="rb") as f: # load in the known encodings 
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location) # loads in the image to be analysed 

    # detect faces in the image and get their encodings
    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    # used to display the image 
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    unkown_names = []

    # looped so multiple faces within the image can be detected
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name: # unkown faces are marked accordingly 
            name = "Unknown"
            unkown_names.append(1)
        else: 
            unkown_names.append(0)

        _display_face(draw, bounding_box, name) # used to generate display with labelled bounding boxes over faces 
    
    del draw
    if display: 
        pillow_image.show() # displays the resultant image 

    return unkown_names


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
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR) # draw bounding box 
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    ) # label bounding box 


# run on the validation images to detect the faces within those unseen images 
def validate(model: str = "hog"):

    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            names = recognize_faces(
                    image_location=str(filepath.absolute()), model=model
                )


# An alarm is set off when only unknown faces are found within an image
def unkown_alarm(imgPath, model: str = "hog", display: bool = True):

    filepath = Path(imgPath)
    if filepath.is_file():
        names = recognize_faces(image_location=str(filepath.absolute()), 
                                model=model, 
                                display = display)
        if all(names) == True: # checks if all detected faces are unknown  
            print("ALARM!")
    

#encode_known_faces() # run to train model on faces within the trianing folder 
#recognize_faces("validation\img3.jpg") # test recognition function on a single image
#recognize_faces("test.jpg")
#validate() # run recognition function on all validation images 
#unkown_alarm("validation\img1.jpg") # alarm check is carried out on passed image 
#pass `display = False` to not display the image during the check