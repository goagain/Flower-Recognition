import tensorflow as tf
import cv2
import numpy as np
import tkinter
from sklearn.preprocessing import LabelEncoder

new_model = tf.keras.models.load_model('deployment/model.h5')

print(new_model.summary())

IMG_SIZE=128

cap = cv2.VideoCapture('deployment/videoplayback.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

subtitlePosX = width // 2 - 200

subtitlePosY = height - 100

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

videoWriter = cv2.VideoWriter('deployment/videoplayback_output.mp4', fourcc, fps, (width, height))

print(f'{fps=}, {width=}, {height=}')

encoder = LabelEncoder()
encoder.classes_ = np.load('deployment/label_encoder.npy')

def make_subtitle(Y):
    y = Y[0]
    argmax = np.argmax(y)
    label = encoder.inverse_transform([argmax])
    return f'{label[0]}: {y[argmax]:.3f}'

frameID = 0
lastSubtitle = ''
while cap.isOpened():
    ok, frame = cap.read()
    if ok:
        if frameID % 10 == 0:
            print(frameID)
            img = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))
            X = np.array([np.array(img) / 255])
            Y = new_model.predict(X)
            lastSubtitle = make_subtitle(Y)
        cv2.putText(frame, lastSubtitle, (subtitlePosX, subtitlePosY), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
        videoWriter.write(frame)
        frameID+=1
    else:
        videoWriter.release()
        break