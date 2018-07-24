import os
import numpy as np
import dlib
import cv2

from data import FaceData

class FaceDetector:
    _detector = None
    _predictor = None

    def detect(self, image, downSampleRatio = None):
        if FaceDetector._detector is None or FaceDetector._predictor is None:
            FaceDetector._detector = dlib.get_frontal_face_detector()

            faceModel = os.path.abspath('{}/models/face_model.dat' \
                            .format(os.path.dirname(__file__)))
            FaceDetector._predictor = dlib.shape_predictor(faceModel)

        detImage = image

        detectedFaces = FaceDetector._detector(detImage, 1)
        if len(detectedFaces) == 0:
            return False, None

        region = detectedFaces[0]
        faceShape = FaceDetector._predictor(image, region)
        face = FaceData()
        face.landmarks = np.array([[p.x, p.y] for p in faceShape.parts()])
        margin = 10
        x, y, w, h = cv2.boundingRect(face.landmarks)
        face.region = (
                       max(x - margin, 0),
                       max(y - margin, 0),
                       min(x + w + margin, image.shape[1] - 1),
                       min(y + h + margin, image.shape[0] - 1)
                      )

        return True, face