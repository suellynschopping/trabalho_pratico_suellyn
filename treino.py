from ultralytics import YOLO

def print_metricas(model, particao):
    if particao not in ('val', 'test'):
        print('particao invalida')
        return

    metricas = model.val(data=path_config_yaml, verbose=False, split=particao)
    print('\n\n-----------------------')
    print('Metricas particao', particao)
    print('-----------------------')
    print(f'   mAP@0.50          : {metricas.box.map50:0.3f}')
    print(f'   mAP@0.50:0.95     : {metricas.box.map:0.3f}')
    print(f'   Precisao media    : {metricas.box.mp:0.3f}')
    print(f'   Recall medio      : {metricas.box.mr:0.3f}')
    print(60*'-')
    for i, cls_idx in enumerate(metricas.box.ap_class_index):
        nome = model.names[cls_idx]
        ap   = metricas.box.ap50[i]
        print(f'  {int(cls_idx)} {nome:<15} AP@50 = {ap:.4f}')

if __name__ == '__main__':
    # instancia modelo nano pre-treinado — melhor para datasets pequenos
    yolo_custom = YOLO("yolov8n.pt")

    # dados identificacao do projeto/modelo
    projeto = "transfer_v3_ep100"
    nome_modelo = "yolo_transfer_n"
    path_config_yaml = 'config.yaml'

    # treino com configuracoes otimizadas para dataset pequeno:
    # - yolov8n: menos parametros = menos overfitting com poucos dados
    # - freeze=10: congela backbone (features do COCO) e treina so a cabeca
    #   de deteccao — estrategia classica de transfer learning para poucos dados
    # - 100 epocas / patience 40: mais tempo para convergir
    # - lr0=0.005: learning rate intermediario — mais estavel que 0.01
    # - augmentacao variada: rotacao, shear, perspectiva para mais diversidade
    # - negativos ja incluidos em train/images sem labels correspondentes
    results_treino = yolo_custom.train(
        data=path_config_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        device='cuda:0',
        project=projeto,
        name=nome_modelo,
        exist_ok=False,
        patience=40,
        plots=True,
        amp=False,
        verbose=False,
        freeze=10,
        lr0=0.005,
        lrf=0.01,
        close_mosaic=20,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        shear=5.0,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.2,
    )

    best_model_path = f'runs/detect/{projeto}/{nome_modelo}/weights/best.pt'
    # instancia melhor epoca modelo treinado
    yolo_best = YOLO(best_model_path)

    print_metricas(yolo_best, 'val')
    print_metricas(yolo_best, 'test')
