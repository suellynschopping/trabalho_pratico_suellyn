"""
Script de inferencia - Deteccao de Armas (Pistola / Fuzil)
Uso:
    python inferencia.py                        # roda nas imagens de teste do dataset
    python inferencia.py --source caminho/img.jpg
    python inferencia.py --source caminho/video.mp4
    python inferencia.py --source caminho/pasta/
"""
import argparse
from ultralytics import YOLO

# Caminho padrao do melhor modelo treinado
DEFAULT_MODEL = 'runs/detect/transfer_v4_ep100_full/yolo_transfer_n/weights/best.pt'
DEFAULT_SOURCE = 'dataset/test/images'

def main():
    parser = argparse.ArgumentParser(description='Inferencia YOLOv8 - Deteccao de Armas')
    parser.add_argument('--model',  default=DEFAULT_MODEL,  help='Caminho para o arquivo .pt do modelo')
    parser.add_argument('--source', default=DEFAULT_SOURCE, help='Imagem, video ou pasta de entrada')
    parser.add_argument('--conf',   type=float, default=0.25, help='Limiar de confianca (default: 0.25)')
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        save_txt=False,
        project='runs/inferencia',
        name='predicoes',
        exist_ok=True,
    )

    print(f'\nInferencia concluida!')
    print(f'Resultados salvos em: runs/inferencia/predicoes/')
    print(f'Total de itens processados: {len(results)}')

if __name__ == '__main__':
    main()
