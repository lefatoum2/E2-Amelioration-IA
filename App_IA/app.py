import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

import streamlit as st
import cv2
import torch
from PIL import Image

model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='Models/best.pt', force_reload=True)
classifier = tf.keras.models.load_model('C:/App_IA/Models/classifier_r4.h5')
classifier.load_weights('C:/App_IA/Models/best_class_4.h5')
l_c = ['Close', 'Entre-ouverte', 'Ouverte']

def cache_video(value:str)->None:
    """
    Create a cv2.VideoCapture object kept in cache to give streamlit the ability to call 
    the release function
    
    """

    if 'capture' in st.session_state.keys():
        st.session_state['capture'].release()

    st.session_state['capture'] = cv2.VideoCapture(value)


def image_traitement(Img, classifier):
    Label = [' Porte ']    
    font = cv2.FONT_HERSHEY_PLAIN
    colors = (235, 82, 82)
    boxes = []
    classid = 0

    CONFIDENCE_THRESHOLD = 0.7
    input = cv2.cvtColor(Img,cv2.COLOR_BGR2RGB)
    results = model_yolo(input)

    IMG_SIZE = 140


    for i in range(0,len(results.pred[0])) :
        if results.pred[0][i,4] > CONFIDENCE_THRESHOLD :
            
            x = int(results.pred[0][i,0])
            y = int(results.pred[0][i,1])
            w = int(results.pred[0][i,2])
            h = int(results.pred[0][i,3])
            box = np.array([x, y, w, h])
            boxes.append(box)
    
    for box in boxes:
        color = colors
        crop = [(box[0], box[1]), (box[1]+box[2], box[1]+box[3])]
        
        crop_img = input[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0]]
        new_array = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
        x_test = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        x_test = x_test.astype('float')/255

        classication = classifier.predict(x_test)
        y_pred = np.argmax(classication,axis=1)
        name = f' {l_c[y_pred[0]]} '
        probability = f' {round(classication.max()*100, 2)} %'
        cv2.rectangle(Img, (box[0],box[1]), (box[1]+box[2],box[1]+box[3]), color, 2)
        cv2.rectangle(Img, (box[0], box[1]), (box[1] + box[2], box[1]+20), color, -1)
        cv2.putText(Img, Label[classid]+name+probability, (box[0], box[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

    return Img    

st.title(":door: Interface de détection de portes")
st.write("""Vous pouvez sélectionner une image ou une vidéo afin d'utiliser les modèles IA de détection, vous pouvez 
aussi détecter les portes en temps réel en activant la webcam.""")

selectbox = st.selectbox("Choissisez :", ["Image", "Video", "Camera"])
if selectbox == "Image":
    uploaded_file = st.file_uploader("Choisissez une image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image.convert('RGB'))
        image = image_traitement(image, classifier)    
        
        st.image(image)

# Create columns for start:stop buttons

if selectbox == "Camera" or selectbox == "Video":
    col1,col2,col3 = st.columns(3)
    with col1:
        start_button = st.button("Start",key='start_button')

    with col2:
        stop_button = st.button("Stop",key='stop_button')


    # Init placeholders to display video and read plates data
    stframe = st.empty()

    # Start inference for choosen stream
    if start_button:

        if selectbox == "Camera":
            output = 0
            scale_cam = .5
            weight = 1

        if selectbox == "Video":
            output = "Exemple/Vidéo_Porte.MOV"
            scale_cam = 2.5
            weight = 5

        cache_video(output)
        predictions = []
        colors = (235, 82, 82)

        while st.session_state['capture'].isOpened():
            ret, frame = st.session_state['capture'].read()
            if selectbox == "Video":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else :
                pass
            # if frame is read correctly ret is True
            if not ret:
                print("Aucune frame reçue...")
                break

            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result = model_yolo(image)
            result_coord =  result.pandas().xyxy[0].values
            color = colors

            
            if result_coord is not None:
                 
                for pred in result_coord:

                    x1 = int(pred[0])
                    y1 = int(pred[1])
                    x2 = int(pred[2])
                    y2 = int(pred[3])

                    start = (x1,y1)
                    end = (x2,y2)

                    name = pred[-1]
                    color = (0, 0, 255)
                    crop_img = image[start[1]:end[1], start[0]:end[0]]
                    image_r = cv2.rectangle(image, start, end, color, 4)

                    new_array = cv2.resize(crop_img, (140, 140)) 
                    x_test = np.array(new_array).reshape(-1, 140, 140, 3)
                    x_test = x_test.astype('float')/255

                    classication = classifier.predict(x_test)
                    y_pred = np.argmax(classication,axis=1)
                    state = f' {l_c[y_pred[0]]} '
                    probability = f' {round(classication.max()*100, 2)} %'
                    image = cv2.putText(image_r, name+state+probability, (x1,y1-20), cv2.FONT_HERSHEY_SIMPLEX, scale_cam, color, weight, cv2.LINE_AA)

                    stframe.image(image)

            if stop_button:
                if 'capture' in st.session_state.keys():
                    st.session_state['capture'].release() 


