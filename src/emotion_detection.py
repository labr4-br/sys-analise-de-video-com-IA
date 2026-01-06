from fer import FER
import cv2
import numpy as np

class EmotionDetection:
    """
    Detecta emoções nas faces detectadas
    Retorna lista de emoções: [(x, y, w, h, confidence), ...]

    Args:
        frame: Frame BGR do OpenCV
        faces: Lista de tuplas (x, y, w, h, confidence) com as faces detectadas
    """
    def __init__(self):
        # FER sem detector de faces (mtcnn=False) - usaremos apenas para análise de emoções
        self.detector = FER(mtcnn=False)

    def detect_emotions_from_faces(self, frame, faces):
        """
        Detecta emoções nas faces detectadas
        Retorna lista de dicionários com informações das faces e emoções
        
        Args:
            frame: Frame original
            faces: Lista de tuplas (x, y, w, h, confidence) do MediaPipe
            
        Returns:
            Lista de dicionários: [{"x": x, "y": y, "w": w, "h": h, "confidence": conf, "emotion": emotion_type, "emotion_score": score}, ...]
        """
        faces_with_emotions = []
        
        for face_data in faces:
            x, y, w, h, confidence = face_data
            
            # Adicionar margem para capturar mais contexto facial
            margin = 0.2
            x_margin = int(w * margin)
            y_margin = int(h * margin)
            
            # Calcular coordenadas com margem
            x1 = max(0, x - x_margin)
            y1 = max(0, y - y_margin)
            x2 = min(frame.shape[1], x + w + x_margin)
            y2 = min(frame.shape[0], y + h + y_margin)
            
            # Extrair região da face
            face_region = frame[y1:y2, x1:x2]
            
            # Verificar se a região é válida
            if face_region.size == 0 or face_region.shape[0] < 10 or face_region.shape[1] < 10:
                continue
            
            # Detectar emoções na região recortada
            result = self.detector.detect_emotions(face_region)
            
            # Desenhar retângulo da face (verde)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Se encontrou emoções na região
            if result and len(result) > 0:
                emotions = result[0]["emotions"]
                
                # Find the emotion with the highest score
                emotion_type = max(emotions, key=emotions.get)
                emotion_score = emotions[emotion_type]
                
                # Display the emotion type and its confidence level
                emotion_text = f"{emotion_type}: {emotion_score:.2f}"
                cv2.putText(frame, emotion_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Adicionar à lista de retorno
                faces_with_emotions.append({
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "confidence": confidence,
                    "emotion": emotion_type,
                    "emotion_score": emotion_score
                })
            else:
                # Se não detectou emoção, mostrar apenas a confiança da detecção de face
                cv2.putText(frame, f"Face: {confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Adicionar à lista mesmo sem emoção detectada
                faces_with_emotions.append({
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "confidence": confidence,
                    "emotion": "desconhecida",
                    "emotion_score": 0.0
                })
        
        return faces_with_emotions

