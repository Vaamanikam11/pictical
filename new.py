import numpy as np 
import pandas as pd
#import tensorflow as tf 
import cvlib as cv 
import cv2 
import time

from imageai.Detection import ObjectDetection

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=("nature.jpg"), output_image_path=("imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

