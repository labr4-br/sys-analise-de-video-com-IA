# Tech Challenge 4 - Análise de Vídeo com IA

Sistema de análise de vídeo que utiliza visão computacional e processamento de imagens para detectar rostos, classificar emoções faciais e identificar atividades em vídeos.

## Descrição

Este projeto implementa uma solução completa de análise de vídeo que combina múltiplas técnicas de visão computacional para:

- **Detecção Facial**: Identifica rostos em cada frame usando MediaPipe Face Detection e Haar Cascade (como fallback)
- **Classificação de Emoções**: Analisa expressões faciais e classifica emoções como alegre, triste, surpreso, pensativo, entre outras
- **Detecção de Atividades**: Monitora o movimento global do vídeo e classifica em níveis de atividade (parado, movimento leve, moderado, intenso)
- **Geração de Resumo**: Cria relatórios automáticos com estatísticas detalhadas e métricas de qualidade

## Funcionalidades

### 1. Detecção Facial
- Utiliza **MediaPipe Face Detection** como método principal
- Fallback automático para **Haar Cascade** quando necessário
- Suporta detecção de múltiplos rostos por frame
- Detecta rostos frontais e de lado
- Calcula confiança de detecção para cada rosto identificado

### 2. Classificação de Emoções
O sistema identifica as seguintes emoções:
- **Alegre/Sorridente**: Boca aberta com intensidade alta
- **Triste**: Boca fechada com intensidade baixa
- **Surpreso**: Boca e olhos muito abertos
- **Pensativo**: Olhos baixos, boca fechada, pouca variação
- **Desdém**: Sobrancelhas assimétricas, boca fechada
- **Careta**: Assimetria significativa da boca
- **Angústia**: Boca parcialmente aberta, intensidade média
- **Neutro**: Expressão padrão
- **Rosto de Lado**: Quando o rosto não está frontal

### 3. Detecção de Atividades
Classifica o movimento global do vídeo em:
- **Parado**: Pouco ou nenhum movimento (< 3)
- **Movimento Leve**: Movimento sutil (3-8)
- **Movimento Moderado**: Movimento médio (8-20)
- **Movimento Intenso**: Movimento significativo (> 20)

### 4. Geração de Relatórios
O sistema gera dois tipos de relatórios:
- **Relatório em Texto** (`resumo_automatico.txt`): Resumo legível com estatísticas
- **Relatório JSON** (`resumo_automatico_detalhado.json`): Dados estruturados para análise posterior

## Dependências

O projeto utiliza as seguintes bibliotecas Python:

- `numpy==1.24.3` - Operações numéricas e arrays
- `opencv-python==4.8.1.78` - Processamento de imagens e vídeo
- `mediapipe==0.10.7` - Detecção facial e análise de landmarks
- `protobuf==3.20.3` - Serialização de dados (requerido pelo MediaPipe)

## Instalação

1. Clone o repositório ou navegue até o diretório do projeto
2. Crie um ambiente virtual (recomendado)
3. Instale as dependências descritas em requirements.txt

## Estrutura do Projeto

```
tech_challenge_4_pos_tech_ia/
├── src/
│   ├── main.py                 # Script principal de execução
│   ├── face_emotion.py         # Módulo de detecção facial e emoções
│   ├── activity_detection.py   # Módulo de detecção de atividades
│   ├── summary.py              # Módulo de geração de resumos
│   ├── haarcascade_frontalface_default.xml  # Classificador Haar Cascade
│   └── haarcascade_smile.xml   # Classificador adicional
├── outputs/                    # Diretório de saída (criado automaticamente)
│   ├── annotated_video.mp4     # Vídeo processado com anotações
│   ├── resumo_automatico.txt   # Relatório em texto
│   └── resumo_automatico_detalhado.json  # Relatório JSON
├── requirements.txt            # Dependências do projeto
├── video_tech.mp4              # Vídeo de exemplo (se disponível)
└── README.md                   # Este arquivo
```

## Como Usar

### Execução Básica

Execute o script principal com o caminho do vídeo:

```bash
python src/main.py --video_path video_tech.mp4
```

### Parâmetros

- `--video_path`: Caminho para o arquivo de vídeo a ser processado (padrão: `video_tech.mp4`)

### Exemplo

```bash
python src/main.py --video_path meu_video.mp4
```

## Saídas do Sistema

Após o processamento, o sistema gera:

1. **Vídeo Anotado** (`outputs/annotated_video.mp4`):
   - Vídeo com bounding boxes coloridos ao redor dos rostos
   - Labels de emoção para cada rosto detectado
   - Informações de atividade no canto superior
   - Informações de debug (abertura da boca, olhos, etc.)

2. **Relatório de Resumo** (`outputs/resumo_automatico.txt`):
   - Estatísticas gerais (total de frames, data/hora)
   - Métricas de qualidade da detecção
   - Distribuição de atividades
   - Distribuição de emoções
   - Transições emocionais mais frequentes
   - Análise temporal
   - Recomendações técnicas

3. **Relatório JSON** (`outputs/resumo_automatico_detalhado.json`):
   - Dados estruturados para análise programática
   - Todas as métricas em formato JSON

## Métricas de Qualidade

O sistema calcula e reporta:

- **Taxa de Detecção Facial**: Percentual de frames com rostos detectados
- **Confiança Média de Detecção**: Confiança média das detecções
- **Qualidade Média dos Rostos**: Métrica combinada de tamanho e confiança
- **Estabilidade Emocional**: Medida de consistência das emoções detectadas
- **Distribuição de Métodos**: Uso de MediaPipe vs Haar Cascade
- **Duração Média das Emoções**: Tempo médio que cada emoção persiste

## Cores das Anotações

Cada emoção é representada por uma cor específica no vídeo anotado:

- 🔵 Azul: Surpreso
- 🟢 Verde: Alegre/Sorridente/Neutro
- 🟡 Amarelo: Careta/Pensativo
- 🟣 Magenta: Triste
- 🟠 Laranja: Desdém
- 🟣 Roxo: Angústia
- ⚪ Cinza: Rosto de Lado

## Detalhes Técnicos

### Detecção Facial

O sistema utiliza uma abordagem híbrida:
1. **MediaPipe Face Detection**: Método principal, mais preciso e rápido
2. **Haar Cascade**: Fallback quando MediaPipe não detecta rostos
3. **MediaPipe Face Mesh**: Para análise detalhada de landmarks faciais

### Classificação de Emoções

A classificação utiliza múltiplas métricas:
- Abertura da boca (distância entre lábios)
- Abertura dos olhos
- Posição das sobrancelhas
- Assimetria facial
- Intensidade média e desvio padrão da imagem
- Orientação do rosto (frontal vs lateral)

### Detecção de Atividades

Baseada na diferença absoluta entre frames consecutivos:
- Converte frames para escala de cinza
- Calcula diferença pixel a pixel
- Classifica baseado em limiares empíricos

## Estatísticas Reportadas

O sistema fornece estatísticas detalhadas incluindo:

- Total de frames processados
- Frames com/sem rostos detectados
- Taxa de detecção facial
- Total de rostos detectados
- Distribuição de métodos de detecção (MediaPipe vs Haar)
- Mudanças de emoção detectadas
- Atividades mais frequentes
- Emoções mais frequentes
- Principais transições emocionais
- Análise temporal (amostras)

## Configuração e Ajustes

### Ajustar Limiares de Atividade

Edite `src/activity_detection.py` para modificar os limiares:
```python
if motion_value < 3:
    activity = "parado"
elif motion_value < 8:
    activity = "movimento leve"
# ... etc
```

### Ajustar Sensibilidade de Emoções

Edite `src/face_emotion.py` na função `classify_emotion_with_mesh()` para modificar os limiares de classificação.

### Configurar MediaPipe

Ajuste os parâmetros de detecção em `src/face_emotion.py`:
```python
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 = curto alcance, 1 = longo alcance
    min_detection_confidence=0.5  # Limiar de confiança
)
```

## 📄 Licença

Este projeto foi desenvolvido para o Tech Challenge 4 - Pós-Tech IA.

