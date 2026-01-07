import cv2
import argparse
import os
import sys

# Adicione o diretório atual ao path para importar módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from face_emotion import process_faces_and_emotions, get_detection_stats, reset_detection_stats
    from activity_detection import ActivityDetector
    from summary import SummaryCollector
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Verifique se todos os arquivos estão na mesma pasta:")
    print("- face_emotion.py")
    print("- activity_detection.py")
    print("- summary.py")
    sys.exit(1)


def main(video_path):
    # Resetar estatísticas antes de começar
    reset_detection_stats()
    
    if not os.path.exists(video_path):
        print(f"Vídeo não encontrado em: {video_path}")
        print(f"Diretório atual: {os.getcwd()}")
        print(f"Arquivos no diretório: {os.listdir('.')}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 20.0  # Valor padrão se não conseguir obter FPS
    
    print(f"Processando vídeo: {video_path}")
    print(f"FPS: {fps}")

    activity_detector = ActivityDetector()
    summary = SummaryCollector()

    frame_index = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    os.makedirs("outputs", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        
        if frame_index % 30 == 0:
            print(f"Processando frame {frame_index}...")

        # 1) Reconhecimento facial + 2) Emoções
        faces_info, frame_with_faces = process_faces_and_emotions(frame)

        # 3) Detecção de atividades (nível global do vídeo)
        activity_label, motion_value = activity_detector.update(frame)

        # Atualiza o resumo (contagem de emoções e atividades)
        summary.update(
            frame_index=frame_index,
            faces_info=faces_info,
            activity_label=activity_label,
        )

        # Desenha info de atividade no frame
        text = f"Atividade: {activity_label}"
        cv2.putText(
            frame_with_faces,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Inicializa writer do vídeo de saída
        if out is None:
            h, w, _ = frame.shape
            out_path = "outputs/annotated_video.mp4"
            out = cv2.VideoWriter(
                out_path,
                fourcc,
                fps,
                (w, h),
            )
            print(f"Salvando vídeo anotado em: {out_path}")

        out.write(frame_with_faces)

        # Opcional: mostrar o frame durante o processamento (para debug)
        # cv2.imshow('Processamento', frame_with_faces)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # 4) Geração de resumo automático
    summary_path = "outputs/resumo_automatico.txt"
    summary.export(summary_path)

    # Obter estatísticas de detecção
    face_stats = get_detection_stats()
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA!")
    print("="*60)
    print(f"Total de frames processados: {frame_index}")
    print(f"Vídeo anotado salvo em: outputs/annotated_video.mp4")
    print(f"Resumo automático salvo em: {summary_path}")
    
    print("\n📊 ESTATÍSTICAS DE DETECÇÃO FACIAL")
    print("-"*40)
    print(f"Frames totais analisados: {face_stats['total_frames']}")
    print(f"Frames com rostos: {face_stats['frames_with_faces']}")
    print(f"Frames sem rostos: {face_stats['frames_without_faces']}")
    print(f"Taxa de detecção: {face_stats['frames_with_faces']/max(1, face_stats['total_frames']):.1%}")
    print(f"Total de rostos detectados: {face_stats['total_faces_detected']}")
    print(f"Detecções MediaPipe: {face_stats['mediapipe_detections']}")
    print(f"Detecções Haar Cascade: {face_stats['haar_detections']}")
    print(f"Mudanças de emoção detectadas: {face_stats['emotion_changes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        default="video_tech.mp4",
        help="Caminho para o arquivo de vídeo de entrada.",
    )
    args = parser.parse_args()
    main(args.video_path)