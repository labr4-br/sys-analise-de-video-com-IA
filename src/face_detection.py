import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

import cv2
import numpy as np
from tqdm import tqdm

class MediaPipeFaceDetection:
    """
    Detecta faces no frame usando MediaPipe
    Retorna lista de coordenadas das faces: [(x, y, w, h, confidence), ...]

    Args:
        frame: Frame BGR do OpenCV
        fps: Frames por segundo do vídeo
        
    Returns:
        Lista de tuplas (x, y, w, h, confidence) com as faces detectadas
    """
    def __init__(self):
        """
        Inicializa o detector MediaPipe
        
        Args:
            face_model: Caminho para o modelo MediaPipe
        """
        self.face_model = 'blaze_face_short_range.tflite'
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceDetector = mp.tasks.vision.FaceDetector
        self.FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.frame_timestamp_ms = 0
        self.detector = None
        
    def initialize_detector(self):
        """Inicializa o detector uma única vez"""
        options = self.FaceDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=self.face_model),
            running_mode=self.VisionRunningMode.VIDEO,
            min_detection_confidence=0.6,  # Reduzido para detectar rostos com menor confiança
            min_suppression_threshold=0.3   # Reduzido para permitir múltiplas detecções próximas
        )
        self.detector = self.FaceDetector.create_from_options(options)

    def face_detection(self, frame, fps):
        """
        Detecta faces no frame usando MediaPipe
        Retorna lista de coordenadas das faces: [(x, y, w, h, confidence), ...]

        Args:
            frame: Frame BGR do OpenCV
            fps: Frames por segundo do vídeo
            
        Returns:
            Lista de tuplas (x, y, w, h, confidence) com as faces detectadas
        """
        if self.detector is None:
            self.initialize_detector()
            
        # Pré-processamento para melhorar detecção
        # Equalização de histograma para melhorar contraste
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])
        enhanced_frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
        
        # Converter frame para o formato do MediaPipe
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
        
        # Detectar faces com timestamp crescente
        detection_result = self.detector.detect_for_video(image, self.frame_timestamp_ms)
        
        # Coletar coordenadas das faces detectadas
        faces = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)
            
            # Garantir que as coordenadas estão dentro dos limites
            x = max(0, x)
            y = max(0, y)
            w = min(frame.shape[1] - x, w)
            h = min(frame.shape[0] - y, h)
            
            # Confiança da detecção
            confidence = detection.categories[0].score
            
            faces.append((x, y, w, h, confidence))
        
        # Incrementar timestamp (em milissegundos)
        self.frame_timestamp_ms += int(1000 / fps)
        
        return faces
    
    def close(self):
        """Fecha o detector"""
        if self.detector:
            self.detector.close()


class YOLOFaceDetection:
    def __init__(self, model_path="yolo11n.pt", confidence_threshold=0.6):
        """
        Inicializa o detector YOLO para faces
        
        Args:
            model_path: Caminho para o modelo YOLO (use yolo11n.pt ou yolov8n.pt)
            confidence_threshold: Threshold mínimo de confiança para detecções (0.0 a 1.0)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def face_detection(self, frame):
        """
        Detecta faces no frame usando YOLO
        Retorna lista de coordenadas das faces no mesmo formato do MediaPipe: [(x, y, w, h, confidence), ...]
        
        Args:
            frame: Frame BGR do OpenCV
            
        Returns:
            Lista de tuplas (x, y, w, h, confidence) com as faces detectadas
        """
        # Executar detecção com YOLO (verbose=False para não poluir o console)
        results = self.model(frame, verbose=False)
        
        faces = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
                
            for box in boxes:
                # Obter coordenadas e confiança
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())
                
                # Filtrar por confiança mínima
                if conf < self.confidence_threshold:
                    continue
                
                # Converter de xyxy (x1, y1, x2, y2) para xywh (x, y, width, height)
                x1, y1, x2, y2 = map(int, xyxy)
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                
                # Garantir que as coordenadas estão dentro dos limites do frame
                x = max(0, x)
                y = max(0, y)
                w = min(frame.shape[1] - x, w)
                h = min(frame.shape[0] - y, h)
                
                # Validar dimensões mínimas (evitar detecções muito pequenas)
                if w > 20 and h > 20:
                    faces.append((x, y, w, h, conf))
        
        return faces
    
    def close(self):
        """Libera recursos do modelo YOLO"""
        # YOLO não precisa de close explícito, mas mantemos para consistência
        pass
