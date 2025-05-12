import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    ret,frame = webcam.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)   

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Face Detection',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()









# cv2.CascadeClassifier():This is a function in OpenCV used to load a Haar Cascade classifier





#detectMultiScale():This is the main function in OpenCV's CascadeClassifier class used for object detection
#gray: The input image in grayscale (not color). Haar cascades work better with grayscale images
# scaleFactor:This compensates for any faces that might appear larger or smaller due to the image being zoomed in or out.
#minNeighbors:This controls the sensitivity of the detector.


