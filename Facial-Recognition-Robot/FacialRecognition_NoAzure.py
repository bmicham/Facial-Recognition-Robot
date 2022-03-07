import cv2

CaptureFace = True

cam = cv2.VideoCapture(0)
faceC = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

while (CaptureFace):
    img = cam.read()[1]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.equalizeHist(imgGray)

    faces = faceC.detectMultiScale(imgGray,1.1,10)

    for (x,y,w,h) in faces:
        faceRecImg = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        print(x,y)

    
    cv2.imshow("Camera Feed",img)
    cachedImage = cv2.imwrite("CapturedImage.jpeg", img)

    if cv2.waitKey(1) == 32:
        break

