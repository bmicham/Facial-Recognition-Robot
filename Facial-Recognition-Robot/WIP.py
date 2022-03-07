import pyttsx3
import face_recognition
from cv2 import cv2
import numpy as np
import glob
from pathlib import Path
import time
import maestro

engine = pyttsx3.init()

#region variables
voices = engine.getProperty('voices')         
engine.setProperty('voice', voices[1].id)   
rate = engine.getProperty('rate')   
engine.setProperty('rate', 135)

video_capture = cv2.VideoCapture(0)
known_face_names = []

facesLocation = glob.glob('faces\*.jpg')
known_face_encodings = {"Person Name":[], "Face Encoding":[]}

servo = maestro.Controller('COM7')
servo.setTarget(5,6000)
servo.setAccel(5,2)
servo.setSpeed(5,1500)
servoPosition = 6000

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
no_face = True
DoDebug = True
HasBeenGreeted = False
#endregion

for file in facesLocation:
    person_face = face_recognition.load_image_file(file)
    person_face_encoding = face_recognition.face_encodings(person_face, model='small')[0]
    known_face_encodings["Person Name"].append(Path(file).stem)
    known_face_encodings["Face Encoding"].append(person_face_encoding)
    known_face_names.append(Path(file).stem)
    print('Adding ' + Path(file).stem + ' to known faces')

def DetectAndGreet(name):
    if HasBeenGreeted == False:
        if name == "Unknown":
            engine.say("I'm sorry but I dont recognize your face")
            engine.runAndWait()
            no_face = False
            HasBeenGreeted == True
        else:
            engine.say('Hello ' + name + ' how are you?')
            engine.runAndWait()
            no_face = False
            HasBeenGreeted == True
        time.sleep(30)

while no_face == True:
    ret, frame = video_capture.read()
    rows, cols, _ = frame.shape

    frameCenter = int(cols / 2)
    faceCenter = int(cols / 2)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings['Face Encoding'], face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings['Face Encoding'], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        right *= 4
        left *= 4

        #DetectAndGreet(name)

        if DoDebug:
            center = (int(frameCenter), int(rows / 2))
            centerofFace = (int((left + right) / 2), int(rows / 2))

            cv2.circle(frame, center, 5, (0, 255, 0), 2)
            cv2.circle(frame, centerofFace, 5, (0, 255, 0), 2)

        faceCenter = int((left + right) / 2)

        if faceCenter < frameCenter -30:
            servoPosition += 350
        elif faceCenter > frameCenter + 30:
            servoPosition -= 350

        print(servoPosition)
        if servoPosition <= 7000 or servoPosition >= 4000:
            servo.setTarget(5, servoPosition)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break