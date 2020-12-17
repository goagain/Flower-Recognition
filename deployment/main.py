from typing import Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder
import getopt
import sys

import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model


class Processor:
    def __init__(self) -> None:
        super().__init__()
        self.__model = load_model('./model.h5')

        self.IMG_SIZE = 128
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load('./label_encoder.npy')

    def process_video(self, video_path, output):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            subtitlePosX = width // 2 - 200

            subtitlePosY = height - 100

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            videoWriter = cv2.VideoWriter(output, fourcc, fps, (width, height))

            # print(f'fps = {fps}, width = {width}, height = {height}')

            frameID = 0
            lastSubtitle = ''
            while cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    if frameID % 10 == 0:
                        print(frameID)
                        label, probability = self.predict(frame)
                        lastSubtitle = f'{label}: {probability:.3f}'
                    cv2.putText(frame, lastSubtitle, (subtitlePosX, subtitlePosY),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
                    videoWriter.write(frame)
                    frameID += 1
                else:
                    break

        finally:
            videoWriter.release()
            return

    def predict(self, img) -> Tuple[str, float]:
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        X = np.array([np.array(img) / 255])
        Y = self.__model.predict(X)

        y = Y[0]
        argmax = np.argmax(y)
        label = self.encoder.inverse_transform([argmax])
        return (label[0], y[argmax])

    def process_img(self, img_path) -> str:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        return self.predict(img)


def usage():
    print(
        """
Usage:sys.args[0] [option]
-h or --help：Show Help
-i or --input：input file path
-o or --output: output file path
-t or --type: type of media, available values are [img, video], default is img
-v or --version：Show version
"""
    )


if len(sys.argv) == 1:
    usage()
    sys.exit()
if __name__ == "__main__":
    output = 'output.mp4'
    input = None
    typ = 'img'

    try:
        if len(sys.argv) == 1:
            usage()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], "ho:i:t:bv", [
                                   "help", "output=", "input=", "type=", "version"])
        for cmd, arg in opts:
            if cmd in ("-h", "--help"):
                usage()
                sys.exit()
            elif cmd in ("-o", "--output"):
                output = arg
            elif cmd in ("-i", "--input"):
                input = arg
            elif cmd in ("-t", "--type"):
                if arg == 'img' or arg == 'video':
                    typ = arg
                else:
                    raise getopt.GetoptError(type=arg)
            elif cmd in ("-v", "--version"):
                print("{0} version 1.0".format(sys.argv[0]))
    except getopt.GetoptError:
        print("argv error,please input")
    if input == None:
        print("an input path must be specified!", file=sys.stderr)
        exit(-1)
    processor = Processor()
    if typ == 'img':
        print(processor.process_img(input))
    else:
        processor.process_video(input, output)
