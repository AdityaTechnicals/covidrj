import os
import cv2
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from keras.models import model_from_json
from keras.preprocessing import image
from django.core.files.storage import FileSystemStorage


# Create your views here.

covid_pred = ['Covid-19', 'Non Covid-19']
IMAGE_SIZE = 64
resnet_model = 'predictor/model_weights/ResNet50/ResNet50_Model.hdf5'
resnet_json = 'predictor/model_weights/ResNet50/ResNet50_Model.json'

def read_image(filepath):
    return cv2.imread(filepath) 

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

def clear_mediadir():
    media_dir = "./media"
    for f in os.listdir(media_dir):
        os.remove(os.path.join(media_dir, f))

def index(request):
    if request.method == "POST" :
        clear_mediadir() 
        
        img = request.FILES['ImgFile']

        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        img_path = fs.path(filename)

        pred_arr = np.zeros(
            (1, IMAGE_SIZE, IMAGE_SIZE, 3))

        im = read_image(img_path)
        if im is not None:
            pred_arr[0] = resize_image(im, (IMAGE_SIZE, IMAGE_SIZE))
        
        pred_arr = pred_arr/255
        

        resnet_start = time.time()
        with open(resnet_json, 'r') as resnetjson:
            resnetmodel = model_from_json(resnetjson.read())

        resnetmodel.load_weights(resnet_model)
        label_resnet = resnetmodel.predict(pred_arr)
        idx_resnet = np.argmax(label_resnet[0])
        cf_score_resnet = np.amax(label_resnet[0])
        resnet_end = time.time()

        resnet_exec = resnet_end - resnet_start

        print('Prediction (ResNet50): ', covid_pred[idx_resnet])
        print('Confidence Score (ResNet50) : ',cf_score_resnet)
        print('Prediction Time (ResNet50) : ', resnet_exec)
        print(img_path)

        response = {}
        response['table'] = "table"
        response['col0'] = " "
        response['col1'] = "ResNet50"
        response['row1'] = "Results"
        response['row2'] = "Confidence Score"
        response['row3'] = "Prediction Time (s)"
        response['r_pred'] = covid_pred[idx_resnet]
        response['r_cf'] = cf_score_resnet
        response['r_time'] = resnet_exec
        response['image'] = "../media/" + img.name
        return render(request, 'index.html', response)
    else:
        return render(request, 'index.html')
