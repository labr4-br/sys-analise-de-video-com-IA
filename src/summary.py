import os
import numpy as np
from collections import defaultdict
from datetime import datetime
import json

class SummaryCollector:
    def __init__(self):
        self.total_frames = 0
        self.activity_counts = defaultdict(int)
        self.emotion_counts = defaultdict(int)
        self.emotion_per_frame = []
        self.face_sizes = []
        self.detection_confidences = []
        self.detection_methods = defaultdict(int)
        self.frame_face_counts = []  # Número de rostos por frame
        self.emotion_transitions = defaultdict(int)
        self.last_emotion_per_face = {}
        
        # Novas métricas
        self.emotion_durations = defaultdict(list)
        self.current_emotion_start = {}
        self.face_qualities = []
        self.temporal_analysis = []

    def update(self, frame_index, faces_info, activity_label):
        """Atualiza estatísticas com informações do frame atual"""
        self.total_frames = frame_index
        self.activity_counts[activity_label] += 1
        
        # Contar emoções
        face_count = len(faces_info)
        self.frame_face_counts.append(face_count)
        
        for i, face_info in enumerate(faces_info):
            emotion = face_info.get("emotion", "desconhecido")
            self.emotion_counts[emotion] += 1
            
            # Rastrear duração das emoções por rosto
            face_id = i  # Simplificado - em produção usar tracking ID
            if face_id not in self.current_emotion_start:
                self.current_emotion_start[face_id] = (emotion, frame_index)
            else:
                last_emotion, start_frame = self.current_emotion_start[face_id]
                if last_emotion != emotion:
                    # Registra duração da emoção anterior
                    duration = frame_index - start_frame
                    self.emotion_durations[last_emotion].append(duration)
                    self.emotion_transitions[f"{last_emotion}->{emotion}"] += 1
                    self.current_emotion_start[face_id] = (emotion, frame_index)
            
            # Coletar métricas de qualidade
            if "detection_confidence" in face_info:
                self.detection_confidences.append(face_info["detection_confidence"])
            
            if "detection_method" in face_info:
                self.detection_methods[face_info["detection_method"]] += 1
            
            if "face_area" in face_info:
                self.face_sizes.append(face_info["face_area"])
                
                # Calcular qualidade baseada em tamanho e confiança
                area = face_info["face_area"]
                conf = face_info.get("detection_confidence", 0.5)
                quality = min(1.0, (area / 10000) * conf)  # Normalizado
                self.face_qualities.append(quality)
        
        # Análise temporal (amostrar a cada 30 frames)
        if frame_index % 30 == 0:
            self.temporal_analysis.append({
                "frame": frame_index,
                "face_count": face_count,
                "emotions": [f.get("emotion", "desconhecido") for f in faces_info],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

    def calculate_metrics(self):
        """Calcula métricas de qualidade"""
        metrics = {}
        
        # Taxa de detecção
        frames_with_faces = sum(1 for count in self.frame_face_counts if count > 0)
        metrics["face_detection_rate"] = frames_with_faces / max(1, self.total_frames) if self.total_frames > 0 else 0
        
        # Média de rostos por frame
        metrics["avg_faces_per_frame"] = np.mean(self.frame_face_counts) if self.frame_face_counts else 0
        
        # Qualidade de detecção
        metrics["avg_detection_confidence"] = np.mean(self.detection_confidences) if self.detection_confidences else 0
        metrics["avg_face_quality"] = np.mean(self.face_qualities) if self.face_qualities else 0
        
        # Distribuição de tamanhos
        if self.face_sizes:
            metrics["avg_face_size"] = np.mean(self.face_sizes)
            metrics["min_face_size"] = np.min(self.face_sizes)
            metrics["max_face_size"] = np.max(self.face_sizes)
        else:
            metrics["avg_face_size"] = 0
            metrics["min_face_size"] = 0
            metrics["max_face_size"] = 0
        
        # Duração média das emoções
        metrics["avg_emotion_duration"] = {}
        for emotion, durations in self.emotion_durations.items():
            if durations:
                metrics["avg_emotion_duration"][emotion] = np.mean(durations)
        
        # Estabilidade emocional (menos transições = mais estável)
        total_faces = sum(self.emotion_counts.values())
        total_transitions = sum(self.emotion_transitions.values())
        metrics["emotional_stability"] = 1 - (total_transitions / max(1, total_faces)) if total_faces > 0 else 0
        
        # Métodos de detecção usados
        metrics["detection_method_distribution"] = dict(self.detection_methods)
        
        return metrics

    def export(self, output_path="outputs/resumo_automatico.txt"):
        """Exporta o resumo com métricas de qualidade"""
        # Garantir que o diretório existe
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calcular métricas
        quality_metrics = self.calculate_metrics()
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("RESUMO AUTOMÁTICO DA ANÁLISE DE VÍDEO\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📊 INFORMAÇÕES GERAIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total de frames analisados: {self.total_frames}\n")
            f.write(f"Data/hora da análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            
            f.write("🎯 MÉTRICAS DE QUALIDADE DA DETECÇÃO\n")
            f.write("-" * 40 + "\n")
            f.write(f"Taxa de detecção facial: {quality_metrics['face_detection_rate']:.1%}\n")
            f.write(f"Média de rostos por frame: {quality_metrics['avg_faces_per_frame']:.2f}\n")
            f.write(f"Confiança média de detecção: {quality_metrics['avg_detection_confidence']:.2f}/1.0\n")
            f.write(f"Qualidade média dos rostos: {quality_metrics['avg_face_quality']:.2f}/1.0\n")
            
            if quality_metrics['avg_face_size'] > 0:
                f.write(f"Tamanho médio dos rostos: {quality_metrics['avg_face_size']:.0f} pixels\n")
                f.write(f"Tamanho mínimo: {quality_metrics['min_face_size']:.0f} pixels\n")
                f.write(f"Tamanho máximo: {quality_metrics['max_face_size']:.0f} pixels\n")
            else:
                f.write("Informações de tamanho dos rostos não disponíveis\n")
            
            f.write(f"Estabilidade emocional: {quality_metrics['emotional_stability']:.1%}\n\n")
            
            f.write("🔄 DISTRIBUIÇÃO DOS MÉTODOS DE DETECÇÃO\n")
            f.write("-" * 40 + "\n")
            detection_methods = quality_metrics.get('detection_method_distribution', {})
            if detection_methods:
                total_detections = sum(detection_methods.values())
                for method, count in detection_methods.items():
                    percentage = count / total_detections * 100 if total_detections > 0 else 0
                    f.write(f"- {method}: {count} detecções ({percentage:.1f}%)\n")
            else:
                f.write("Informações de métodos de detecção não disponíveis\n")
            f.write("\n")
            
            f.write("🚶 ATIVIDADES MAIS FREQUENTES\n")
            f.write("-" * 40 + "\n")
            if self.activity_counts:
                sorted_activities = sorted(self.activity_counts.items(), 
                                          key=lambda x: x[1], reverse=True)
                for activity, count in sorted_activities:
                    percentage = count / self.total_frames * 100 if self.total_frames > 0 else 0
                    f.write(f"- {activity}: {count} frames ({percentage:.1f}%)\n")
            else:
                f.write("Nenhuma atividade registrada\n")
            f.write("\n")
            
            f.write("😊 EMOÇÕES MAIS FREQUENTES\n")
            f.write("-" * 40 + "\n")
            total_faces = sum(self.emotion_counts.values())
            if total_faces > 0:
                sorted_emotions = sorted(self.emotion_counts.items(), 
                                        key=lambda x: x[1], reverse=True)
                
                for emotion, count in sorted_emotions:
                    percentage = count / total_faces * 100
                    
                    # Adicionar duração média se disponível
                    duration_info = ""
                    if emotion in quality_metrics['avg_emotion_duration']:
                        avg_dur = quality_metrics['avg_emotion_duration'][emotion]
                        duration_info = f" (dura {avg_dur:.1f} frames em média)"
                    
                    f.write(f"- {emotion}: {count} rostos detectados ({percentage:.1f}%){duration_info}\n")
            else:
                f.write("Nenhuma emoção detectada\n")
            f.write("\n")
            
            f.write("🔄 PRINCIPAIS TRANSIÇÕES EMOCIONAIS\n")
            f.write("-" * 40 + "\n")
            if self.emotion_transitions:
                sorted_transitions = sorted(self.emotion_transitions.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]  # Top 10
                for transition, count in sorted_transitions:
                    f.write(f"- {transition}: {count} vezes\n")
            else:
                f.write("Nenhuma transição significativa detectada.\n")
            f.write("\n")
            
            f.write("📈 ANÁLISE TEMPORAL (amostras)\n")
            f.write("-" * 40 + "\n")
            if self.temporal_analysis:
                for sample in self.temporal_analysis[:5]:  # Mostrar primeiras 5 amostras
                    f.write(f"Frame {sample['frame']} ({sample['timestamp']}): ")
                    f.write(f"{sample['face_count']} rosto(s) - ")
                    f.write(f"Emoções: {', '.join(sample['emotions'])}\n")
            else:
                f.write("Análise temporal não disponível\n")
            f.write("\n")
            
            f.write("💡 RECOMENDAÇÕES TÉCNICAS\n")
            f.write("-" * 40 + "\n")
            
            # Análise automática baseada nas métricas
            recommendations = []
            
            if quality_metrics['face_detection_rate'] < 0.3:
                recommendations.append("⚠️  Taxa de detecção baixa. Verifique:")
                recommendations.append("   • Iluminação do ambiente")
                recommendations.append("   • Posicionamento da câmera")
                recommendations.append("   • Oclusão dos rostos")
            
            if quality_metrics['avg_detection_confidence'] < 0.4:
                recommendations.append("⚠️  Confiança de detecção abaixo do ideal.")
                recommendations.append("   • Considere ajustar os limiares de detecção")
                recommendations.append("   • Melhorar a qualidade do vídeo")
            
            if quality_metrics['emotional_stability'] < 0.5:
                recommendations.append("⚠️  Baixa estabilidade emocional detectada.")
                recommendations.append("   • Pode indicar mudanças rápidas de expressão")
                recommendations.append("   • Ou instabilidade na detecção")
            
            if quality_metrics.get('avg_face_size', 0) < 2000 and quality_metrics['avg_face_size'] > 0:
                recommendations.append("⚠️  Rostos muito pequenos na imagem.")
                recommendations.append("   • Aproxime a câmera dos sujeitos")
                recommendations.append("   • Use zoom digital se disponível")
            
            if recommendations:
                for rec in recommendations:
                    f.write(rec + "\n")
            else:
                f.write("✅ Todas as métricas estão dentro dos parâmetros ideais\n")
            
            f.write("\n✅ CONFIGURAÇÃO IDEAL:\n")
            f.write("   • Taxa de detecção: > 70%\n")
            f.write("   • Confiança média: > 0.6\n")
            f.write("   • Tamanho do rosto: > 4000 pixels\n")
            f.write("   • Estabilidade emocional: > 60%\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("=" * 60 + "\n")
        
        print(f"Resumo salvo em: {output_path}")
        
        # Também salvar como JSON para análise posterior
        json_path = output_path.replace(".txt", "_detalhado.json")
        detailed_data = {
            "geral": {
                "total_frames": self.total_frames,
                "timestamp": datetime.now().isoformat()
            },
            "atividades": dict(self.activity_counts),
            "emocoes": dict(self.emotion_counts),
            "metricas_qualidade": quality_metrics,
            "transicoes": dict(self.emotion_transitions),
            "analise_temporal": self.temporal_analysis
        }
        
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)
            print(f"Relatório detalhado (JSON) salvo em: {json_path}")
        except Exception as e:
            print(f"Erro ao salvar JSON: {e}")