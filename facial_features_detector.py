import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, col):
    faces = face.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(col, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_col = col[y:y+h, x:x+w]
        smiles = smile.detectMultiScale(roi_gray, 1.8, 20)
        for(sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_col, (sx, sy), (sx+sw, sy+sh), (0,255,0), 2)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_col, (ex,ey), (ex+ew, ey+eh), (0, 0, 255), 2)
    return col


video_cap = cv2.VideoCapture(0)

while True:
    _, frame = video_cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = detect(gray_img, frame)
    cv2.imshow('result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()