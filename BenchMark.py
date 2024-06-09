import pandas as pd
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
import cv2
import time
from statistics import mean
from evaluator_modfied import inference_on_dataset
from detectron2 import model_zoo

cfg = get_cfg()
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6  # Ajuste o limite NMS conforme necessário
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # Ajuste o limite de confiança conforme necessário
cfg.MODEL.ROI_HEADS.NUM_DETECTIONS = 1  # Limite uma detecção por imagem
cfg.MODEL.DEVICE = 'cpu'

def getBenchMark(model_path, data_path):
    try:
        cfg.MODEL.WEIGHTS = model_path
        predictor = DefaultPredictor(cfg)
    except:
        return Exception
    time_log = []

    for img in os.listdir(data_path):
        img_path = os.path.join(data_path,img)
        img_load = cv2.imread(img_path)
        time_inference_start = time.perf_counter()
        predictor(img_load)
        time_inference_end = time.perf_counter()
        time_log.append(time_inference_end - time_inference_start)
        print(f"{time_inference_end - time_inference_start} S")

    data_final = {'inference_min': min(time_log), 'inference_max': max(time_log), 'inference_mean': mean(time_log)}
    data_final = pd.DataFrame([data_final])

    model_name = model_path.split('/')[-3]
    directory = f"runs/detectron/benchmark/{model_name}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    return data_final.to_csv(f"{directory}/results.csv", index=False)


def benchmarkModelsInFolder(models_folder, data_path):
    models = os.listdir(models_folder)
    for model in models:
        print(f"Model : {model}")
        getBenchMark(f"{models_folder}/{model}/model_final.pth", data_path)
    return None


print(benchmarkModelsInFolder("models","CsiLab-BrainTumor-Detection-3/teste"))