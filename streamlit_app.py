# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 07:20:54 2024

@author: rwest
"""
import cv2
import numpy as np
import streamlit as st

scaler = 'minmaxscaler2.pkl'
knn = 'knn_model.pkl'



#Image function
def image_recog(image):
    # image = cv2.GaussianBlur(image,(11,11),0)
    image = image[60:420, 100:540] #cropping outer edges
    # st.image(image, caption="Captured Image", use_column_width=True)     #-----Image cropped and blurred
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # adaptive Thresholding works better for thin digits.
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
    # _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # weird alternative, works sometimes
    image = cv2.bitwise_not(image)
    st.image(image, caption="Image after preprocessing", use_column_width=True)
    image = image.flatten()
    image = image.reshape(1, -1)
    image = scaler.transform(image)
    pred = knn.predict(image)
    st.write("Predicted digit:", pred)
    



#Function to capture image from webcam
def capture_image(camera):
    _, frame = camera.read()
    if frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        
        return st.write("No frame found")



#Main function
def main():
    st.title("Webcam Live Feed for image recognition")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    capture_button = st.button("Capture Image")
    st.write("Requirements: keep your digit centered. Digit must have clear outlines. Less shadows gives better result.")
    
    while camera.isOpened():
                    _, frame = camera.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame)
    if capture_button:
        captured_frame = capture_image(camera)
        image_recog(captured_frame)

    else:
            st.write('Stopped')







if __name__ == "__main__":
    main()
    
    
#streamlit run "LOCAL FILEPATH TO RUN STREAMLIT"











