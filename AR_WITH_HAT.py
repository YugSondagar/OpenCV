import cv2
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

hat_image = cv2.imread(os.path.join('.', 'data', 'hat.jpeg'))


webcap = cv2.VideoCapture(0)

while True:
    ret, frame = webcap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        hat_width = w
        hat_height = int(hat_image.shape[0] * (hat_width / hat_image.shape[1]))
        resized_hat = cv2.resize(hat_image, (hat_width, hat_height))

        hat_x = x
        hat_y = y - hat_height

        if hat_y < 0:
            continue

        if hat_y + hat_height > frame.shape[0] or hat_x + hat_width > frame.shape[1]:
            continue

        try:
            frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width] = resized_hat
        except:
            pass

    cv2.imshow('Hat on Face (JPEG)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcap.release()
cv2.destroyAllWindows()
