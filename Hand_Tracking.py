import cv2

webcam = cv2.VideoCapture(0)

lower_skin = (0, 20, 70)
upper_skin = (20, 255, 255)


while True:
    ret, frame = webcam.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    skin_mask = cv2.inRange(hsv_frame,lower_skin,upper_skin)

    skin_segement = cv2.bitwise_and(frame,frame,mask=skin_mask)

    gray = cv2.cvtColor(skin_segement, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    contours,__builtins__ = cv2.findContours(blurred, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            x,y,w,h = cv2.boundingRect(cnt)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()