import cv2
import numpy as np


class ActivityDetector:
    """
    Classifica a atividade global do vídeo em:
    - parado
    - movimento leve
    - movimento moderado
    - movimento intenso
    usando a diferença entre frames.
    """

    def __init__(self):
        self.prev_gray = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return "desconhecida", 0.0

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        motion_value = float(np.mean(diff))

        # Regras simples de limiares – você pode ajustar empiricamente
        if motion_value < 3:
            activity = "parado"
        elif motion_value < 8:
            activity = "movimento leve"
        elif motion_value < 20:
            activity = "movimento moderado"
        else:
            activity = "movimento intenso"

        return activity, motion_value