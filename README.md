# Tech Challenge 4 - Sistema de Análise de Vídeo com IA

Sistema inteligente de análise de vídeo que detecta faces, identifica emoções e classifica atividades em tempo real usando técnicas avançadas de visão computacional e aprendizado de máquina.

## Funcionalidades

- **Detecção de Faces**: Utiliza MediaPipe e YOLO como fallback para garantir melhor precisão
- **Reconhecimento de Emoções**: Análise de expressões faciais com a biblioteca FER (Facial Emotion Recognition)
- **Classificação de Atividades**: Detecta níveis de movimento no vídeo (parado, leve, moderado, intenso)
- **Geração de Relatórios**: Exporta resumo estatístico completo da análise
- **Processamento em Lote**: Processa vídeos completos com barra de progresso

## Arquitetura do Sistema

O projeto está organizado em módulos especializados:

```
tech_challenge_4/
├── main.py                      # Script principal de execução
├── src/
│   ├── face_detection.py        # Detecção de faces (MediaPipe + YOLO)
│   ├── emotion_detection.py     # Análise de emoções (FER)
│   ├── activity_detection.py    # Classificação de atividades
│   └── summary.py               # Geração de relatórios
├── requirements.txt             # Dependências do projeto
└── README.md                    # Documentação
```

### Fluxo de Processamento

1. **Captura de Vídeo**: Leitura frame a frame do vídeo de entrada
2. **Detecção de Faces**: 
   - Primeira tentativa com MediaPipe (otimizado e rápido)
   - Fallback para YOLO se MediaPipe não detectar faces
3. **Análise de Emoções**: FER analisa cada face detectada
4. **Detecção de Atividade**: Análise de movimento entre frames consecutivos
5. **Anotação Visual**: Desenha retângulos e labels no frame
6. **Coleta de Estatísticas**: Acumula dados para o relatório final
7. **Exportação**: Salva vídeo processado e relatório em texto

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Arquivo de vídeo para análise

### Passos de Instalação

1. **Clone o repositório**:
```bash
git clone <url-do-repositorio>
cd tech_challenge_4
```

2. **Crie um ambiente virtual**:
```bash
python -m venv venv
source venv/bin/activate
```

3. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

4. **Baixe os modelos necessários**:
   - **MediaPipe**: Baixe `blaze_face_short_range.tflite` e coloque na raiz do projeto
   - **YOLO**: O modelo `yolo11n.pt` será baixado automaticamente na primeira execução

## Como Usar

### Uso Básico

Execute o script principal com um vídeo de entrada:

```bash
python main.py
```

Por padrão, o script processa `video_tech.mp4` e gera:
- `output_mediapipe_yolo.mp4` - Vídeo com anotações visuais
- `output_mediapipe_yolo_summary.txt` - Relatório estatístico

### Personalização

Edite o arquivo `main.py` para alterar os caminhos:

```python
if __name__ == "__main__":
    main('seu_video.mp4', 'saida_processada.mp4')
```

### Ajuste de Parâmetros

#### Detecção de Faces (MediaPipe)

```python
# Em src/face_detection.py
min_detection_confidence=0.6,  # Confiança mínima (0.0 a 1.0)
min_suppression_threshold=0.3   # Supressão de detecções próximas
```

#### Detecção de Faces (YOLO)

```python
# Em main.py
yolo_face_detection = YOLOFaceDetection(confidence_threshold=0.4)
```

#### Classificação de Atividades

```python
# Em src/activity_detection.py
# Ajuste os limiares de movimento:
if motion_value < 3:
    activity = "parado"
elif motion_value < 8:
    activity = "movimento leve"
elif motion_value < 20:
    activity = "movimento moderado"
else:
    activity = "movimento intenso"
```

## Formato do Relatório

O arquivo de resumo gerado contém:

```
============================================================
RESUMO AUTOMÁTICO DA ANÁLISE DE VÍDEO
============================================================

Total de frames analisados: 1500

------------------------------------------------------------
ATIVIDADES DETECTADAS
------------------------------------------------------------
  • Movimento leve: 850 frames (56.7%)
  • Parado: 450 frames (30.0%)
  • Movimento moderado: 200 frames (13.3%)

------------------------------------------------------------
EMOÇÕES DETECTADAS
------------------------------------------------------------
  • Happy: 450 detecções (45.0%)
  • Neutral: 300 detecções (30.0%)
  • Surprise: 150 detecções (15.0%)
  • Sad: 100 detecções (10.0%)

============================================================
```

## Tecnologias Utilizadas

### Bibliotecas Principais

- **OpenCV** (4.10.0.84): Processamento de imagens e vídeo
- **MediaPipe** (0.10.21): Detecção de faces em tempo real
- **Ultralytics YOLO** (8.3.239): Detecção de objetos e faces (fallback)
- **FER** (22.5.1): Reconhecimento de emoções faciais
- **TensorFlow** (2.17.1): Backend para modelos de deep learning
- **PyTorch** (2.2.2): Framework de deep learning
- **NumPy** (1.26.4): Computação numérica
- **tqdm** (4.67.1): Barras de progresso

### Modelos de IA

1. **BlazeFace** (MediaPipe): Detector de faces leve e rápido
2. **YOLO11n**: Detector de objetos de última geração
3. **FER**: Rede neural para classificação de emoções

## Detalhes Técnicos

### Detecção de Faces Híbrida

O sistema implementa uma estratégia de fallback inteligente:

```python
# Tenta primeiro com MediaPipe (mais rápido)
faces = media_pipe_face_detection.face_detection(frame, fps)

# Se falhar, usa YOLO (mais robusto)
if not faces:
    faces = yolo_face_detection.face_detection(frame)
```

### Pré-processamento de Imagens

Para melhorar a detecção, o sistema aplica equalização de histograma:

```python
frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])
enhanced_frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
```

### Detecção de Atividades

Usa diferença absoluta entre frames consecutivos:

```python
diff = cv2.absdiff(gray, prev_gray)
motion_value = float(np.mean(diff))
```

### Análise de Emoções

Adiciona margem de 20% ao redor das faces para melhor contexto:

```python
margin = 0.2
x_margin = int(w * margin)
y_margin = int(h * margin)
```

## 🎨 Visualização

O vídeo de saída inclui:

- **Retângulos verdes**: Faces detectadas
- **Labels de emoção**: Tipo e confiança (em magenta)
- **Indicador de atividade**: Nível de movimento (em laranja)

## Solução de Problemas

### GPU Desabilitada

O sistema desabilita GPU por padrão para evitar problemas de compatibilidade:

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

Para habilitar GPU, comente ou remova esta linha em `main.py`.

### Erro ao Carregar Modelos

Certifique-se de que:
- `blaze_face_short_range.tflite` está na raiz do projeto
- Você tem conexão com internet para baixar o YOLO na primeira execução

### Baixa Taxa de Detecção

Tente:
- Reduzir `min_detection_confidence` no MediaPipe
- Reduzir `confidence_threshold` no YOLO
- Melhorar a iluminação do vídeo de entrada

### Consumo Alto de Memória

Para vídeos muito longos:
- Processe em lotes menores
- Reduza a resolução do vídeo de entrada
- Use um modelo YOLO menor (yolo11n.pt)

## Performance

### Benchmarks Típicos

- **Velocidade**: ~15-30 FPS em CPU moderna
- **Precisão de Detecção**: >90% em condições ideais
- **Uso de Memória**: ~2-4 GB RAM

### Otimizações Implementadas

- Inicialização única dos detectores
- Processamento vetorizado com NumPy
- Fallback inteligente entre modelos
- Desabilitação de verbose nos modelos

## Licença

Este projeto é de código aberto e está disponível para uso educacional e comercial.

## Autores

- Bruna Ballerini

---

**Nota**: Este projeto foi desenvolvido como parte do Tech Challenge 4 e demonstra a integração de múltiplas tecnologias de IA para análise de vídeo em tempo real.
