import cv2
import tensorflow as tf
from skimage.feature import hog

MODEL = './model'
IMG_SIZE = 128

model = tf.keras.models.load_model(MODEL)

labels_dict={0: 'Not smiling', 1: 'Smiling'}
color_dict={0: (0, 0, 255), 1: (0, 255, 0)}

camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(frame_gray, 1.1, 10)

    for (x, y, w, h) in faces:
        face = frame_gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        features = hog(face, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=False)
        pred = model.predict(features.reshape(1,-1))
        label = 1 if pred[0] >= 0.5 else 0

        cv2.rectangle(frame, (x,y), (x+w,y+h), color_dict[label], 2)
        cv2.rectangle(frame, (x,y-40), (x+w,y), color_dict[label], -1)
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
    cv2.imshow('SMILE DETECTION', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()