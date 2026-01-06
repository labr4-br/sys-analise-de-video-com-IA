import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Desabilitar GPU para evitar problemas de compilação

import cv2
import numpy as np
from tqdm import tqdm

from src.emotion_detection import EmotionDetection
from src.face_detection import MediaPipeFaceDetection, YOLOFaceDetection
from src.activity_detection import ActivityDetection
from src.summary import SummaryCollector

def main(video_path, output_path):
    if not os.path.exists(video_path):
        print(f"Vídeo não encontrado em: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    media_pipe_face_detection = MediaPipeFaceDetection()
    yolo_face_detection = YOLOFaceDetection(confidence_threshold=0.4)
    emotion_detection = EmotionDetection()
    activity_detection = ActivityDetection()
    summary_collector = SummaryCollector()
    
    frame_index = 0
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        frame_index += 1
        ret, frame = cap.read()
        if not ret:
            break
        
        # MediaPipe detecta as faces e retorna coordenadas
        faces = media_pipe_face_detection.face_detection(frame, fps)

        # Se MediaPipe não detectou faces, tenta com YOLO
        if not faces:
            faces = yolo_face_detection.face_detection(frame)
        
        # FER analisa emoções apenas nas regiões detectadas e retorna faces com emoções
        faces_with_emotions = emotion_detection.detect_emotions_from_faces(frame, faces)

        activity = activity_detection.update(frame)

        # Desenha info de atividade no frame
        text = f"Atividade: {activity}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 165, 0),
            2,
        )

        summary_collector.update(frame_index, faces_with_emotions, activity)
        
        out.write(frame)
    
    # Liberar recursos
    media_pipe_face_detection.close()
    yolo_face_detection.close()
    out.release()
    cap.release()
    
    # Exportar resumo
    summary_path = output_path.replace('.mp4', '_summary.txt')
    summary_collector.export(summary_path)
    print(f"\nResumo exportado para: {summary_path}")
    
if __name__ == "__main__":
    main('video_tech.mp4', 'output_mediapipe_yolo.mp4')