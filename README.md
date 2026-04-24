========================================================================
  TRABALHO PRÁTICO - VISÃO COMPUTACIONAL E MODELOS GENERATIVOS
  Detecção de Armas de Fogo (Pistola e Fuzil) com YOLOv8
========================================================================

Aluna : Suellyn Schopping
Curso : MBA em IA
Disciplina : Sistemas Generativos em Visão Computacional
Repositório: https://github.com/suellynschopping/trabalho_pratico_suellyn


------------------------------------------------------------------------
1. OBJETIVO
------------------------------------------------------------------------

Treinar um modelo de detecção de objetos capaz de identificar e localizar
duas classes de armas de fogo em imagens:

    0 - Pistola
    1 - Fuzil

O projeto utiliza transfer learning a partir do modelo YOLOv8n pré-treinado
no dataset COCO, ajustando apenas a cabeça de detecção ao domínio do
problema (armas de fogo).


------------------------------------------------------------------------
2. ESTRUTURA DO REPOSITÓRIO
------------------------------------------------------------------------

trabalho_pratico_suellyn/
│
├── dataset/                      # Dataset dividido em train / val / test
│                                 # (cada pasta contém images/ e labels/ no
│                                 # formato YOLO)
│
├── images/                       # Imagens auxiliares para testes de
│                                 # inferência
│
├── output/                       # Resultados das inferências (imagens
│                                 # anotadas com as detecções)
│
├── runs/detect/                  # Pasta gerada pela Ultralytics durante
│                                 # o treino (pesos, métricas, gráficos)
│
├── config.yaml                   # Configuração do dataset para o YOLO
│                                 # (paths, splits e nomes das classes)
│
├── classes_dataset.txt           # Lista das classes do dataset
├── ambiente.txt                  # Detalhes do ambiente de execução
├── requirements.txt              # Dependências Python do projeto
│
├── teste_ambiente.py             # Script para validar a instalação
│                                 # (Python, PyTorch, CUDA, Ultralytics)
│
├── treino.py                     # Script principal de treino e avaliação
├── inferencia.py                 # Script de inferência com o modelo treinado
├── teste_modelo_custom.ipynb     # Notebook para testes exploratórios do
│                                 # modelo custom
│
├── relatorio_trabalho_pratico.docx   # Relatório técnico do trabalho
├── trabalho_pratico_final.docx       # Versão final do relatório
└── README.txt                        # Este arquivo


------------------------------------------------------------------------
3. AMBIENTE E DEPENDÊNCIAS
------------------------------------------------------------------------

- Python 3.10+
- PyTorch com suporte a CUDA (treino executado em GPU: device='cuda:0')
- Ultralytics (YOLOv8)
- OpenCV, NumPy, Matplotlib
- Demais dependências listadas em requirements.txt


------------------------------------------------------------------------
4. INSTALAÇÃO
------------------------------------------------------------------------

Passo a passo para reproduzir o ambiente:

    # 1) Clonar o repositório
    git clone https://github.com/suellynschopping/trabalho_pratico_suellyn.git
    cd trabalho_pratico_suellyn

    # 2) Criar e ativar um ambiente virtual (recomendado)
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux / macOS:
    source .venv/bin/activate

    # 3) Instalar dependências
    pip install -r requirements.txt

    # 4) (Opcional) Validar o ambiente
    python teste_ambiente.py


------------------------------------------------------------------------
5. CONFIGURAÇÃO DO DATASET (config.yaml)
------------------------------------------------------------------------

O arquivo config.yaml aponta para a pasta do dataset e define os splits
e as classes:

    path  : <caminho absoluto para a pasta dataset>
    train : train/images
    val   : val/images
    test  : test/images
    names :
      0: Pistola
      1: Fuzil

IMPORTANTE: antes de rodar o treino, ajuste o campo "path" do config.yaml
para o caminho absoluto do dataset na sua máquina.


------------------------------------------------------------------------
6. COMO EXECUTAR
------------------------------------------------------------------------

6.1) Treino do modelo

    python treino.py

    Principais hiperparâmetros (definidos em treino.py):
      - modelo base ........ YOLOv8n (yolov8n.pt)
      - épocas ............. 100 (completas, sem early stopping)
      - imgsz .............. 640
      - batch .............. 8
      - device ............. cuda:0
      - patience ........... 0 (early stopping desabilitado)
      - freeze ............. 10 (congela o backbone - transfer learning)
      - lr0 / lrf .......... 0.005 / 0.01
      - augmentações ....... mosaic, copy_paste, hsv, rotação, shear,
                             flip horizontal, perspectiva

    Ao final do treino, o script avalia o melhor modelo (best.pt) nas
    partições de validação e teste, imprimindo:
      - mAP@0.50
      - mAP@0.50:0.95
      - Precisão média
      - Recall médio
      - AP@50 por classe (Pistola e Fuzil)

    Os pesos do melhor modelo ficam em:
      runs/detect/transfer_v4_ep100_full/yolo_transfer_n/weights/best.pt


6.2) Inferência

    python inferencia.py

    As imagens anotadas com as detecções são salvas na pasta output/.


6.3) Testes exploratórios

    O notebook teste_modelo_custom.ipynb permite carregar o modelo
    treinado e testar inferências de forma interativa, visualizando
    as detecções em imagens específicas.


------------------------------------------------------------------------
7. ABORDAGEM E DECISÕES DE PROJETO
------------------------------------------------------------------------

- Modelo YOLOv8n (nano): escolhido por ter menos parâmetros, o que
  reduz o risco de overfitting em um dataset relativamente pequeno.

- Transfer learning com freeze=10: o backbone pré-treinado no COCO é
  congelado e apenas a cabeça de detecção é ajustada. Essa é uma
  estratégia clássica para cenários com poucos dados rotulados,
  aproveitando features visuais genéricas já aprendidas.

- 100 épocas com patience=40: dá tempo suficiente para a convergência
  e usa early stopping para evitar treino desnecessário.

- lr0=0.005: learning rate intermediário, mais estável do que o padrão
  0.01 para este cenário.

- Augmentações variadas (mosaic, copy_paste, HSV, rotação, shear,
  perspectiva, flip horizontal): aumentam a diversidade efetiva do
  dataset, melhorando a generalização.

- Uso de imagens negativas (sem rótulo correspondente) em train/images
  para reduzir falsos positivos.


------------------------------------------------------------------------
8. RESULTADOS
------------------------------------------------------------------------

As métricas finais (mAP@0.50, mAP@0.50:0.95, precisão, recall e AP por
classe) são impressas no console ao final da execução do treino.py e
também estão disponíveis no relatório técnico:

    relatorio_trabalho_pratico.docx
    trabalho_pratico_final.docx

Gráficos e matrizes de confusão gerados automaticamente pela Ultralytics
ficam em:

    runs/detect/transfer_v4_ep100_full/yolo_transfer_n/


------------------------------------------------------------------------
9. OBSERVAÇÕES
------------------------------------------------------------------------

- O treino foi executado com GPU NVIDIA (device='cuda:0'). Para rodar
  em CPU, basta alterar device='cpu' em treino.py (o treino ficará
  significativamente mais lento).

- Caso a pasta runs/detect/transfer_v4_ep100_full/ já exista, o parâmetro
  exist_ok=False fará o YOLO criar um novo diretório versionado
  (transfer_v4_ep100_full_2, etc.).

- O dataset contém conteúdo sensível (armas de fogo) e foi utilizado
  exclusivamente para fins acadêmicos, com o objetivo de estudar
  técnicas de detecção de objetos em visão computacional.

========================================================================
