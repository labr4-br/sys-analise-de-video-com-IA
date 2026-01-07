import cv2
import os
import numpy as np
import mediapipe as mp

# Para MediaPipe 0.10.7, use esta forma de importar
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Variáveis globais para estatísticas
detection_stats = {
    'total_frames': 0,
    'frames_with_faces': 0,
    'frames_without_faces': 0,
    'total_faces_detected': 0,
    'mediapipe_detections': 0,
    'haar_detections': 0,
    'emotion_changes': 0,
    'last_emotion': None
}

def get_detection_stats():
    """Retorna estatísticas de detecção"""
    return detection_stats.copy()

def reset_detection_stats():
    """Reseta as estatísticas"""
    global detection_stats
    detection_stats = {
        'total_frames': 0,
        'frames_with_faces': 0,
        'frames_without_faces': 0,
        'total_faces_detected': 0,
        'mediapipe_detections': 0,
        'haar_detections': 0,
        'emotion_changes': 0,
        'last_emotion': None
    }

def get_cascade_path(filename: str) -> str:
    local_path = os.path.join(os.path.dirname(__file__), filename)
    cv2_base = os.path.dirname(cv2.__file__)
    candidates = [
        local_path,
        os.path.join(cv2_base, "data", filename),
        os.path.join(cv2_base, "data", "haarcascades", filename),
        os.path.join(cv2_base, "haarcascades", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[INFO] Usando cascade '{filename}' em: {p}")
            return p
    msg = (
        f"Arquivo '{filename}' não encontrado.\n"
        "Verifique se ele está na pasta 'src/' ou ajuste o caminho em get_cascade_path()."
    )
    raise FileNotFoundError(msg)

FACE_CASCADE_PATH = get_cascade_path("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Falha ao carregar o classificador de rosto em: {FACE_CASCADE_PATH}")

# Inicializar os modelos do MediaPipe
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,  # Aumentar para 2 para detectar rostos de lado
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Definir índices de landmarks faciais
LEFT_EYEBROW_IDX = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
UPPER_LIP_POINT = 13
LOWER_LIP_POINT = 14
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Novos índices para melhor detecção de caretas e rostos de lado
NOSE_TIP = 4
CHIN = 152
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MOUTH_CENTER = 0

def landmark_distance(landmarks, i, j, w, h):
    """Calcula distância entre dois landmarks com verificação de None"""
    if landmarks is None or i >= len(landmarks) or j >= len(landmarks):
        return 0.0
    xi, yi = landmarks[i].x * w, landmarks[i].y * h
    xj, yj = landmarks[j].x * w, landmarks[j].y * h
    return float(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))

def landmark_x(landmarks, idx):
    """Retorna coordenada x de um landmark"""
    if landmarks is None or idx >= len(landmarks):
        return 0.0
    return float(landmarks[idx].x)

def landmark_y(landmarks, idx):
    """Retorna coordenada y de um landmark com verificação"""
    if landmarks is None or idx >= len(landmarks):
        return 0.0
    return float(landmarks[idx].y)

def average_y(landmarks, indices):
    """Calcula média de coordenadas y com verificação"""
    if not landmarks or not indices:
        return None
    try:
        vals = [landmarks[i].y for i in indices if i < len(landmarks)]
        return float(np.mean(vals)) if vals else None
    except:
        return None

def average_x(landmarks, indices):
    """Calcula média de coordenadas x com verificação"""
    if not landmarks or not indices:
        return None
    try:
        vals = [landmarks[i].x for i in indices if i < len(landmarks)]
        return float(np.mean(vals)) if vals else None
    except:
        return None

def calculate_face_orientation(landmarks, w, h):
    """Calcula a orientação do rosto (frontal ou de lado)"""
    if landmarks is None or len(landmarks) < 10:
        return "frontal", 0.0
    
    # Calcular diferença horizontal entre olhos e nariz
    left_eye_x = average_x(landmarks, [33, 133, 157, 158, 159])  # Olho esquerdo
    right_eye_x = average_x(landmarks, [362, 386, 387, 388, 263])  # Olho direito
    nose_x = landmark_x(landmarks, NOSE_TIP)
    
    if left_eye_x is None or right_eye_x is None or nose_x is None:
        return "frontal", 0.0
    
    # Calcular simetria facial
    left_dist = abs(nose_x - left_eye_x)
    right_dist = abs(right_eye_x - nose_x)
    
    if left_dist == 0 or right_dist == 0:
        return "frontal", 0.0
    
    symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
    tilt = right_dist - left_dist
    
    if symmetry_ratio < 0.6:
        if tilt > 0:
            return "lado_direito", symmetry_ratio
        else:
            return "lado_esquerdo", symmetry_ratio
    else:
        return "frontal", symmetry_ratio

def calculate_mouth_asymmetry(landmarks):
    """Calcula assimetria da boca para detectar caretas"""
    if landmarks is None or len(landmarks) < 300:
        return 0.0
    
    # Pontos da boca
    left_corner_y = landmark_y(landmarks, LEFT_MOUTH_CORNER)
    right_corner_y = landmark_y(landmarks, RIGHT_MOUTH_CORNER)
    upper_lip_y = landmark_y(landmarks, UPPER_LIP_POINT)
    lower_lip_y = landmark_y(landmarks, LOWER_LIP_POINT)
    
    # Calcular diferenças
    corner_diff = abs(left_corner_y - right_corner_y)
    vertical_open = abs(upper_lip_y - lower_lip_y)
    
    # Normalizar
    if vertical_open == 0:
        return 0.0
    
    return corner_diff / vertical_open

def detect_faces(frame, gray):
    """Detecta rostos usando MediaPipe com fallback para Haar Cascade"""
    h, w, _ = frame.shape
    faces = []
    detection_method = "none"
    
    # Usar MediaPipe Face Detector
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        if face_detector is not None:
            results = face_detector.process(rgb)
            
            if results and hasattr(results, 'detections') and results.detections:
                detection_method = "mediapipe"
                detection_stats['mediapipe_detections'] += 1
                
                for detection in results.detections:
                    if hasattr(detection, 'location_data'):
                        bbox = detection.location_data.relative_bounding_box
                        
                        if bbox:
                            x_min = int(bbox.xmin * w)
                            y_min = int(bbox.ymin * h)
                            bw = int(bbox.width * w)
                            bh = int(bbox.height * h)
                            
                            # Ajustar coordenadas
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            bw = max(1, min(w - x_min, bw))
                            bh = max(1, min(h - y_min, bh))
                            
                            # Adicionar confiança da detecção se disponível
                            confidence = detection.score[0] if hasattr(detection, 'score') else 0.5
                            faces.append((x_min, y_min, bw, bh, confidence, "mediapipe"))
    except Exception as e:
        print(f"Erro no MediaPipe face detection: {e}")
    
    # Fallback para Haar Cascade
    if not faces:
        try:
            haar_faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            if haar_faces is not None and len(haar_faces) > 0:
                detection_method = "haar"
                detection_stats['haar_detections'] += 1
                
                for (x, y, w_f, h_f) in haar_faces:
                    # Estimativa de confiança baseada no tamanho e posição
                    size_confidence = min(1.0, (w_f * h_f) / (h * w) * 10)
                    faces.append((x, y, w_f, h_f, size_confidence, "haar"))
        except Exception as e:
            print(f"Erro no Haar Cascade: {e}")
    
    # Atualizar estatísticas
    detection_stats['total_frames'] += 1
    if faces:
        detection_stats['frames_with_faces'] += 1
        detection_stats['total_faces_detected'] += len(faces)
    else:
        detection_stats['frames_without_faces'] += 1
    
    return faces

def classify_emotion_with_mesh(face_gray, face_color):
    """Classifica emoção usando MediaPipe Face Mesh com lógica refinada"""
    h, w = face_gray.shape[:2]
    mean_intensity = float(np.mean(face_gray))
    std_intensity = float(np.std(face_gray))
    
    try:
        face_rgb = cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB)
        face_rgb.flags.writeable = False
        
        if face_mesh is None:
            debug_info = {
                "mouth_open": None, "eye_open": None, "mean_intensity": mean_intensity,
                "std_intensity": std_intensity, "eye_y": None, "eyebrow_diff": None,
                "mouth_corner_tilt": None, "face_orientation": "frontal",
                "mouth_asymmetry": 0.0
            }
            return None, debug_info
        
        result = face_mesh.process(face_rgb)
        
        # Verificações robustas
        if (not result or not hasattr(result, 'multi_face_landmarks') or 
            not result.multi_face_landmarks or len(result.multi_face_landmarks) == 0):
            
            debug_info = {
                "mouth_open": None, "eye_open": None, "mean_intensity": mean_intensity,
                "std_intensity": std_intensity, "eye_y": None, "eyebrow_diff": None,
                "mouth_corner_tilt": None, "face_orientation": "frontal",
                "mouth_asymmetry": 0.0
            }
            return None, debug_info
        
        landmarks = result.multi_face_landmarks[0].landmark
        
        if landmarks is None or len(landmarks) == 0:
            debug_info = {
                "mouth_open": None, "eye_open": None, "mean_intensity": mean_intensity,
                "std_intensity": std_intensity, "eye_y": None, "eyebrow_diff": None,
                "mouth_corner_tilt": None, "face_orientation": "frontal",
                "mouth_asymmetry": 0.0
            }
            return None, debug_info
        
        # Calcular métricas
        mouth_open_px = landmark_distance(landmarks, UPPER_LIP_POINT, LOWER_LIP_POINT, w, h)
        mouth_open = mouth_open_px / h if h > 0 else 0
        
        left_corner_y = landmark_y(landmarks, LEFT_MOUTH_CORNER)
        right_corner_y = landmark_y(landmarks, RIGHT_MOUTH_CORNER)
        mouth_corner_tilt = abs(left_corner_y - right_corner_y)
        
        left_eye_open = landmark_distance(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, w, h) / h if h > 0 else 0
        right_eye_open = landmark_distance(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, w, h) / h if h > 0 else 0
        eye_open = (left_eye_open + right_eye_open) / 2.0
        
        # Calcular eye_y
        eye_y = None
        try:
            eye_y = (
                landmark_y(landmarks, LEFT_EYE_TOP) +
                landmark_y(landmarks, LEFT_EYE_BOTTOM) +
                landmark_y(landmarks, RIGHT_EYE_TOP) +
                landmark_y(landmarks, RIGHT_EYE_BOTTOM)
            ) / 4.0
        except:
            eye_y = None
        
        # Calcular diferença entre sobrancelhas
        left_eyebrow_y = average_y(landmarks, LEFT_EYEBROW_IDX)
        right_eyebrow_y = average_y(landmarks, RIGHT_EYEBROW_IDX)
        eyebrow_diff = abs(left_eyebrow_y - right_eyebrow_y) if (left_eyebrow_y and right_eyebrow_y) else None
        
        # Calcular orientação do rosto
        face_orientation, symmetry_ratio = calculate_face_orientation(landmarks, w, h)
        
        # Calcular assimetria da boca
        mouth_asymmetry = calculate_mouth_asymmetry(landmarks)
        
        # LÓGICA DE CLASSIFICAÇÃO REFINADA
        emotion = "neutro"
        
        # 1. Primeiro verificar se é rosto de lado
        if face_orientation in ["lado_esquerdo", "lado_direito"]:
            emotion = "rosto_lado"
        
        # 2. Verificar surpresa (boca e olhos muito abertos)
        elif mouth_open > 0.08 and eye_open > 0.045:
            emotion = "surpreso"
        
        # 3. Verificar careta (assimetria da boca significativa)
        elif mouth_asymmetry > 0.15 and mouth_open > 0.03:
            emotion = "careta"
        
        # 4. Verificar desdém (sobrancelhas assimétricas, boca fechada)
        elif eyebrow_diff and eyebrow_diff > 0.035 and mouth_open < 0.035:
            emotion = "desdém"
        
        # 5. Verificar angústia (boca parcialmente aberta, intensidade média)
        elif (0.04 <= mouth_open <= 0.07 and 
              60 <= mean_intensity <= 110 and 
              std_intensity > 35 and 
              eye_open < 0.04):
            emotion = "angústia"
        
        # 6. Verificar alegre/sorridente
        elif mouth_open > 0.05:
            if mean_intensity > 95:
                emotion = "sorridente"
            else:
                emotion = "alegre"
        
        # 7. Verificar triste
        elif mouth_open < 0.035 and mean_intensity < 75:
            emotion = "triste"
        
        # 8. Verificar pensativo (olhos baixos, boca fechada, pouca variação)
        elif (mouth_open < 0.035 and 
              70 <= mean_intensity <= 125 and 
              std_intensity < 35 and 
              eye_y and eye_y > 0.52 and 
              eye_open < 0.035):
            emotion = "pensativo"
        
        # 9. Default para neutro

        debug_info = {
            "mouth_open": mouth_open,
            "eye_open": eye_open,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "eye_y": eye_y,
            "eyebrow_diff": eyebrow_diff,
            "mouth_corner_tilt": mouth_corner_tilt,
            "face_orientation": face_orientation,
            "mouth_asymmetry": mouth_asymmetry,
            "symmetry_ratio": symmetry_ratio
        }

        return emotion, debug_info
        
    except Exception as e:
        print(f"Erro no classify_emotion_with_mesh: {e}")
        debug_info = {
            "mouth_open": None, "eye_open": None, "mean_intensity": mean_intensity,
            "std_intensity": std_intensity, "eye_y": None, "eyebrow_diff": None,
            "mouth_corner_tilt": None, "face_orientation": "frontal",
            "mouth_asymmetry": 0.0
        }
        return None, debug_info

def fallback_emotion(face_gray):
    """Classificação de fallback baseada apenas na intensidade da imagem"""
    try:
        if face_gray.size == 0:
            return "neutro"
        
        mean_intensity = float(np.mean(face_gray))
        std_intensity = float(np.std(face_gray))
        
        if mean_intensity < 65:
            return "triste"
        elif 65 <= mean_intensity <= 120 and std_intensity < 25:
            return "pensativo"
        else:
            return "neutro"
    except Exception as e:
        print(f"Erro no fallback_emotion: {e}")
        return "neutro"

def process_faces_and_emotions(frame):
    """Processa o frame para detecção facial e classificação de emoções"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_data = detect_faces(frame, gray)  # Agora retorna mais informações
        faces_info = []
        annotated_frame = frame.copy()
        
        for face_data in faces_data:
            if len(face_data) == 6:
                x, y, w, h, confidence, method = face_data
            else:
                # Para compatibilidade com versão anterior
                x, y, w, h = face_data
                confidence = 0.5
                method = "unknown"
            
            # Verificar se as coordenadas são válidas
            if (w <= 0 or h <= 0 or 
                x >= frame.shape[1] or y >= frame.shape[0] or
                x + w <= 0 or y + h <= 0):
                continue
            
            # Ajustar coordenadas
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            # Extrair regiões do rosto
            try:
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]
                
                if face_gray.size == 0 or face_color.size == 0:
                    emotion = fallback_emotion(face_gray)
                    dbg = None
                else:
                    emotion, dbg = classify_emotion_with_mesh(face_gray, face_color)
                    if emotion is None:
                        emotion = fallback_emotion(face_gray)
            except Exception as e:
                print(f"Erro ao extrair regiões faciais: {e}")
                emotion = "neutro"
                dbg = None
            
            # Rastrear mudanças de emoção
            if detection_stats['last_emotion'] and detection_stats['last_emotion'] != emotion:
                detection_stats['emotion_changes'] += 1
            detection_stats['last_emotion'] = emotion
            
            faces_info.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "emotion": emotion,
                "debug": dbg,
                "detection_confidence": confidence,
                "detection_method": method,
                "face_area": w * h,
                "face_ratio": w / h if h > 0 else 0
            })
            
            # Desenhar bounding box com cor baseada na emoção
            color_map = {
                "surpreso": (255, 0, 0),      # Azul
                "alegre": (0, 255, 0),        # Verde
                "sorridente": (0, 200, 100),  # Verde claro
                "triste": (255, 0, 255),      # Magenta
                "pensativo": (255, 255, 0),   # Ciano
                "desdém": (0, 165, 255),      # Laranja
                "careta": (0, 255, 255),      # Amarelo
                "angústia": (128, 0, 128),    # Roxo
                "rosto_lado": (128, 128, 128),# Cinza
                "neutro": (0, 255, 0)         # Verde
            }
            
            color = color_map.get(emotion, (0, 255, 0))
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
            
            # Adicionar texto da emoção com confiança
            text_y = max(y - 10, 10)
            emotion_text = f"{emotion} ({confidence:.1f})"
            cv2.putText(annotated_frame, emotion_text, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Adicionar informações de debug se disponíveis
            if dbg and dbg.get("mouth_open") is not None:
                debug_lines = [
                    f"mouth:{dbg['mouth_open']:.3f}",
                    f"eye:{dbg.get('eye_open', 0):.3f}",
                    f"mean:{dbg['mean_intensity']:.1f}",
                    f"ori:{dbg.get('face_orientation', 'frontal')}"
                ]
                dy = 13
                for i, line in enumerate(debug_lines):
                    text_y_pos = min(y + h + 15 + i * dy, frame.shape[0] - 10)
                    cv2.putText(annotated_frame, line, 
                               (x, text_y_pos), 
                               cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 255), 1)
        
        return faces_info, annotated_frame
        
    except Exception as e:
        print(f"Erro em process_faces_and_emotions: {e}")
        return [], frame.copy()

# Função para limpar recursos
def cleanup():
    try:
        if 'face_detector' in globals() and face_detector:
            face_detector.close()
        if 'face_mesh' in globals() and face_mesh:
            face_mesh.close()
    except:
        pass