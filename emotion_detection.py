import sys
import argparse
import cv2
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

from faces import FaceDetector
from data import FaceData
from gabor import GaborBank
from emotions import EmotionsDetector

class VideoData:
    def __init__(self):
        self._faceDet = FaceDetector()
        self._bank = GaborBank()
        self._emotionsDet = EmotionsDetector()
        self._face = FaceData()
        self._emotions = OrderedDict()

    def detect(self, frame):
        ret, face = self._faceDet.detect(frame)
        if ret:
            self._face = face
            frame, face = face.crop(frame)
            responses = self._bank.filter(frame)
            self._emotions = self._emotionsDet.detect(face, responses)
            return [True,self._emotions]
        else:
            self._face = None
            return [False]

def main(argv, frameStart, frameEnd):
    video = cv2.VideoCapture(argv[0])
    if not video.isOpened():
        print('Error opening video file {}'.format(argv[0]))
        sys.exit(-1)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    sourceName = argv[0]
    data = VideoData()

    paused = False
    emotions = {}

    #skipping the required number of frames
    count = 0
    while frameStart!=0 and (count+1)!=frameStart:
        ret, img = video.read()
        if not ret:
            return "na"
        count += 1
    
    while True:
        if not paused:
            start = datetime.now()

        ret, img = video.read()
        if ret:
            frame = img.copy()
        else:
            paused = True

        emoti = data.detect(frame)

        if emoti[0]:
            print(list(emoti[1].items()))
            
            for i,j in emoti[1].items():
                if i in emotions:
                    emotions[i] = emotions[i] + j
                else:
                    emotions[i] = j
        else:
            print("Face not detected")

        if paused:
            video.release()
            cv2.destroyAllWindows()
            if emotions!={}:
                return emotions
            return "na"
        else:
            end = datetime.now()
            delta = (end - start)
            if fps != 0:
                delay = int(max(1, ((1 / fps) - delta.total_seconds()) * 1000))
            else:
                delay = 1

            key = cv2.waitKey(delay)

    video.release()
    print(emotions)
    cv2.destroyAllWindows()


def process(video,frameStart, frameEnd):
    emotions = main(video, frameStart, frameEnd)
    if emotions=="na":
        return "na"
    print("\n\n\nEmotion dictionary is :: ", emotions)
    rev_emotions = {j:i for i,j in emotions.items()}
    print("Emotion detected from the video is :: ",rev_emotions[sorted(rev_emotions)[-1]])
    return rev_emotions[sorted(rev_emotions)[-1]]


if __name__ == '__main__':
    process(sys.argv[1:],0,0)