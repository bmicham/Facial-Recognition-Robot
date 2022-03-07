#Source Library TTS- https://pypi.org/project/pyttsx3/
#Pip Download - pip install pyttsx3

#Source Library Face Rec -https://pypi.org/project/face-recognition/
#Pip Install - Download Visual Studio for C++ (You don't need to code anything in it, the library just needs it to compile)
#            - pip install cmake
#            - pip install face-recognition
#            - pip install opencv-python

import pyttsx3
import face_recognition
from cv2 import cv2
import numpy as np

engine = pyttsx3.init()

#region Voice Setup

#Sound of Voice
voices = engine.getProperty('voices')         
#Set Voice Gender (0 = Male, 1 = Female)
engine.setProperty('voice', voices[1].id)   

#Speaking Rate of Voice
rate = engine.getProperty('rate')   
#Change the Rate (135 Sounds Best)
engine.setProperty('rate', 135)     

#endregion

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

#Second Person Picture
Brett_Face = face_recognition.load_image_file("Brett_Face.jpg") #You need to change the file destination
Brett_face_encoding = face_recognition.face_encodings(Brett_Face)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    Brett_face_encoding
]
known_face_names = [
    "Brett"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
no_face = True

while no_face == True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    #Process the frames
    if process_this_frame:
        #Look for faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            #Check for matches
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    #Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        #Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        #Outline the face with a box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #Create the frame details on the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if name == "Brett":
            engine.say("Hello Brett, how are you.")
            engine.runAndWait()
            no_face = False

        elif name == "Unknown":
            engine.say("I'm sorry but I dont recognize your face")
            engine.runAndWait()
            no_face = False

    #Show the camera feed
    cv2.imshow('Video', frame)

    #Press Q to break loop and kill camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#Turn off the webcam
video_capture.release()
cv2.destroyAllWindows()
