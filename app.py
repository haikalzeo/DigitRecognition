import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import pickle

st.title("Drawable Digit Recognition using Support Vector Machine")
st.markdown("> This app demonstrates Digit Recognition using SVM. The app allows users draw digits on to a canvas and an image with prediction digits will be shown as the result.")
st.markdown(">Disclaimer : The performance of our SVM Model on Drawable Digit Recognition showed a realistic results")
st.header("🎲 How to Use the Application")
st.markdown(
    """
    * Draw Digits freely as you want! 
    * You can Undo, Redo or Delete your drawing with button in the bottom left of the canvas 
    * Please Don't forget to Press ⬇️ Send Button in the bottom left of the canvas!
    * Press Predict Button if you're done with your drawing and the result will be shown bellow
    """
    )
st.write(" ")
st.write(" ")
st.write(" ")

st.markdown("Draw Digits [0-9]")
# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)", 
    stroke_width=9,
    stroke_color="black",
    background_color="#eee",
    background_image=None,
    update_streamlit=False,
    height=270,
    width=700,
    drawing_mode="freedraw",
    point_display_radius= 0,
    display_toolbar=True,
    key="full_app",
)

predict = st.button("Predict")

def drawSquare(image):
    '''
    Draws a square around the found digits
    '''		
    b = [0,0,0]
    height, width = image.shape[0], image.shape[1]
    if(height == width): ## if square
      square = image
      return square
    else:
      d_size = cv2.resize(image, (2*width, 2*height), interpolation=cv2.INTER_CUBIC)
      height, width = height * 2, width * 2
      if (height > width):
        padding = (height - width)/2; padding = int(padding)
        d_size_square = cv2.copyMakeBorder(d_size, 0, 0, padding, padding, cv2.BORDER_CONSTANT, None,value=b)
      else:
        padding = (width - height)/2; padding = int(padding)
        d_size_square = cv2.copyMakeBorder(d_size, padding, padding, 0, 0, cv2.BORDER_CONSTANT, None,value=b)

    return d_size_square

def resize(image, dim):
    '''
    Returns orignal image resized to shape 'dim'
    '''	
    b = [0,0,0]	 			
    dim = dim - 4
    squared = image
    r = (float(dim) / squared.shape[1])
    d = (dim, int(squared.shape[0] * r))
    resized = cv2.resize(image, d, interpolation = cv2.INTER_AREA)
    height, width = resized.shape[0], resized.shape[1];
    if (height > width):
      resized = cv2.copyMakeBorder(resized, 0,0,0,1, cv2.BORDER_CONSTANT, value=b)
    if (height < width):
      resized = cv2.copyMakeBorder(resized, 1,0,0,0, cv2.BORDER_CONSTANT, value=b)

    resized = cv2.copyMakeBorder(resized, 2,2,2,2,cv2.BORDER_CONSTANT, value=b)
    height, width = resized.shape[0], resized.shape[1]	
    
    return resized	

file = open('DigitRecognition_SVM_Model.pt', 'rb')
digitRecognition_SVM_Model = pickle.load(file)
file.close()

def recognize_digit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 30, 150)

    contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    contours, _ = zip(*sorted(zip(contours, boundingBoxes), key=lambda b:b[1][0], reverse=False))

    display = []

    for contour in contours:
      (x, y, w, h) = cv2.boundingRect(contour)
        
      if w>= 5 and h>=50:
        area = blur[y:y+h, x:x+w]
        _, area = cv2.threshold(area, 127, 255, cv2.THRESH_BINARY_INV)
            
        new_square = drawSquare(area)
        number = resize(new_square, 20)
        result = number.reshape((1, 400))

        result = result.astype(np.float32)
        res = digitRecognition_SVM_Model.predict(result)
        n = str(int(float(res)))
        display.append(n)

        # draw rectangle around individual digit
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(image, n, (x,y-8), cv2.FONT_ITALIC, 0.7, (0,255,0), 2)

    return image, display

if canvas_result.image_data is not None and predict:
    resImg, resDigit = recognize_digit(canvas_result.image_data)
    st.markdown("Prediction Result")
    st.image(resImg)
    st.text("Digit Prediction : {}".format(resDigit))