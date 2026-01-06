from collections import Counter

class SummaryCollector:
    """
    Coleta e exporta resumo da análise de vídeo
    """
    def __init__(self):
        """
        Inicializa o coletor de resumo

        Args:
            total_frames: Total de frames analisados
            emotion_counter: Contador de emoções
            activity_counter: Contador de atividades
        """
        self.total_frames = 0
        self.emotion_counter = Counter()
        self.activity_counter = Counter()

    def update(self, frame_index, faces_info, activity_label):
        """
        Atualiza o coletor de resumo

        Args:
            frame_index: Índice do frame atual
            faces_info: Lista de tuplas (x, y, w, h, confidence) com as faces detectadas
            activity_label: Label da atividade detectada
        """
        self.total_frames += 1

        # Contagem de emoções dos rostos detectados neste frame
        for face in faces_info:
            self.emotion_counter[face["emotion"]] += 1

        # Contagem de atividades (uma por frame, global)
        self.activity_counter[activity_label] += 1

    def export(self, txt_path):
        """
        Exporta o resumo para um arquivo de texto

        Args:
            txt_path: Caminho para o arquivo de texto
        """
        lines = []
        lines.append("=" * 60)
        lines.append("RESUMO AUTOMÁTICO DA ANÁLISE DE VÍDEO")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Total de frames analisados: {self.total_frames}")
        lines.append("")
        
        lines.append("-" * 60)
        lines.append("ATIVIDADES DETECTADAS")
        lines.append("-" * 60)
        if self.activity_counter:
            total_activity_frames = sum(self.activity_counter.values())
            for act, count in self.activity_counter.most_common():
                percentage = (count / total_activity_frames) * 100
                lines.append(f"  • {act.capitalize()}: {count} frames ({percentage:.1f}%)")
        else:
            lines.append("  • Nenhuma atividade detectada.")
        
        lines.append("")
        lines.append("-" * 60)
        lines.append("EMOÇÕES DETECTADAS")
        lines.append("-" * 60)
        if self.emotion_counter:
            total_emotions = sum(self.emotion_counter.values())
            for emo, count in self.emotion_counter.most_common():
                percentage = (count / total_emotions) * 100
                lines.append(f"  • {emo.capitalize()}: {count} detecções ({percentage:.1f}%)")
        else:
            lines.append("  • Nenhuma emoção detectada.")
        
        lines.append("")
        lines.append("=" * 60)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))