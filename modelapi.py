from typing import List, Dict, Any, Tuple
from fastapi import APIRouter, Depends, status, File, UploadFile, HTTPException, Response
from models import solarFault
from fastapi.responses import JSONResponse
import subprocess
import os
import requests
import shutil

# Model related imports
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

router = APIRouter(
    prefix = "/model",
    tags = ["predeiction"]
)

IMG_HEIGHT = 244
IMG_HEIGHT = 244
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

@router.post('/solar_panel', status_code = status.HTTP_202_ACCEPTED)
async def solar_pannel_fault_predict(request: Dict[str, Any]):
    img_paths = request['img_paths']
    result = []

    for img_path in img_paths:
        # # Load the image and resize it
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        # img = img / 255.0  # Normalize the image
        # img_batch = tf.expand_dims(img_array, axis=0)  # Add a batch dimension

        # Different method of preprocessing the image
        img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Preprocessing the image
        img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)


        # Make prediction
        prediction = solarFault.model.predict(img_preprocessed)
        # prediction = [0, 1]
        score = tf.nn.softmax(prediction[0])  # Get probabilities

        # Find the most likely class
        predicted_class = class_names[np.argmax(score)]
        result.append(predicted_class)
    
    # Removing the images from the directory
    for img_path in img_paths:
        os.remove(img_path)
    os.rmdir('models\\predictImg')
    return JSONResponse({"output" : result})



@router.get('/solar_panel/train', status_code = status.HTTP_200_OK)
async def train_solar_panel_model():
    if os.path.exists('models/solar_panel_fault_detection.keras'):
        os.remove('models/solar_panel_fault_detection.keras')
    subprocess.Popen('python models/solarFault.py', shell=True)
    return JSONResponse({"output" : 'done'})



@router.post('/upload_multiple', status_code = status.HTTP_200_OK)
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    temp_dir = 'models\\predictImg'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    saved_files = []
    for uploaded_file in files:
        file_path = os.path.join(temp_dir, uploaded_file.filename)
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        saved_files.append(file_path)
    
    return JSONResponse({'output' : saved_files})



@router.post('/clear_all', status_code = status.HTTP_200_OK)
def clear_all(downloaded_images: List[str]):
    for img_path in downloaded_images:
        if os.path.exists(img_path):
            os.remove(img_path)
    if os.path.exists('models\\predictImg'):
        os.rmdir('models\\predictImg')