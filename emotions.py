import os
from collections import OrderedDict
import numpy as np

from gabor import GaborBank
from data import FaceData
from faces import FaceDetector

from sklearn import svm
from sklearn.externals import joblib

class InvalidModelException(Exception):
    pass

class EmotionsDetector:
    def __init__(self):
        self._clf = svm.SVC(kernel='rbf', gamma=0.001, C=10,
                            decision_function_shape='ovr',
                            probability=True, class_weight='balanced')

        self._emotions = OrderedDict([
                             (0, 'neutral'), (1, 'happiness'), (2, 'sadness'),
                             (3, 'anger'), (4, 'fear'),  (5, 'surprise'),
                             (6, 'disgust')
                         ])

        modulePath = os.path.dirname(__file__)
        self._modelFile = os.path.abspath('{}/models/emotions_model.dat' \
                            .format(modulePath))

        if not os.path.isfile(self._modelFile):
            raise InvalidModelException('Could not find model file: {}' \
                    .format(self._modelFile))

        if not self.load():
            raise InvalidModelException('Could not load model from file: {}' \
                    .format(self._modelFile))

    def load(self):
        try:
            clf = joblib.load(self._modelFile)
        except:
            return False

        self._clf = clf
        return True

    def _relevantFeatures(self, gaborResponses, facialLandmarks):
        points = np.array(facialLandmarks)
        try:
            responses = gaborResponses[:, points[:, 1], points[:, 0]]
        except:
            w = gaborResponses.shape[2]
            h = gaborResponses.shape[1]

            responses = np.zeros((32, 68), dtype=float)
            for i in range(len(points)):
                x = points[i][0]
                y = points[i][1]
                if x < w and y < h:
                    responses[:, i] = gaborResponses[:, y, x]
                else:
                    responses[:, i] = 0.0

        featureVector = responses.reshape(-1).tolist()

        return featureVector

    def detect(self, face, gaborResponses):
        features = self._relevantFeatures(gaborResponses, face.landmarks)
        return self.predict(features)

    def predict(self, features):
        probas = self._clf.predict_proba([features])[0]
        ret = OrderedDict()
        for i in range(len(self._emotions)):
            label = self._emotions[i]
            ret[label] = probas[i]

        return ret