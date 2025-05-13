import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# hat_image = cv2.imread(os.path.join('.','data','hat.jpeg'),-1)

webcap = cv2.VideoCapture(0)

while True:
    ret, frame = webcap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.circle(frame,(x+w//2,y-20),20,(0,0,255),-1)
    
    cv2.imshow('AR with Object Tracking',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcap.release()
cv2.destroyAllWindows()