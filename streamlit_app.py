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



#image function
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
    



# Function to capture image from webcam
def capture_image(camera):
    _, frame = camera.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



# Main function
def main():
    st.title("Webcam Live Feed for image recognition")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    capture_button = st.button("Capture Image")
    st.write("Requirements: keep your digit centered. Digit must have clear outlines. Less shadows gives better result.")
    
    if capture_button:
        captured_frame = capture_image(camera)
        image_recog(captured_frame)
    while camera.isOpened():
                    _, frame = camera.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame)
    else:
            st.write('Stopped')







if __name__ == "__main__":
    main()
    
    
#streamlit run "c:\users\rwest\desktop\ds23\machine learning\kunskapskontroll_2\streamlit_debug.py"











# "_" = img from camera recieved (True/False), "frame" = actual image, fed into the empty list FRAME_WINDOW
    # if run:
    #     while run:
    #         _, frame = camera.read()
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         FRAME_WINDOW.image(frame)
            
    #         # when Capture Image button is pressed:
    #         if capture_button:
    #             captured_frame = capture_image(camera)
    #             # pred = image_recog(captured_frame)
    #             # st.write("Prediction:", pred)
    #             captured_frame = cv2.GaussianBlur(captured_frame,(33,33), 0)
    #             captured_frame = captured_frame[120:360, 160:480] #cropping outer edges
    #             # (480, 640, 3) webcam reso
    #             captured_frame = cv2.resize(captured_frame, (28,28), interpolation=cv2.INTER_NEAREST)
    #             captured_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
    #             captured_frame = cv2.bitwise_not(captured_frame) #background not black, fix!!
    #             captured_frame = captured_frame.flatten()
    #             captured_frame = captured_frame.reshape(1, -1)
    #             captured_frame = scaler.transform(captured_frame)
    #             pred = knn.predict(captured_frame)
    #             st.write("Predicted digit:", pred)
    #             st.write(captured_frame.shape)
    #             st.write(type(captured_frame))
    #             # st.image(captured_frame, caption="Captured Image", use_column_width=True)
    #             break