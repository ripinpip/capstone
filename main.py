import tensorflow as tf
from tensorflow import keras
import numpy as np
import tkinter
from tkinter import *
from tkinter import ttk
import seaborn as sn
import matplotlib.pyplot as plt
from PIL import Image, EpsImagePlugin
import cv2

# load the dataset
(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()

# scale the images to increase accuracy when trained
xTrain = xTrain / 255
xTest = xTest / 255

# reshape the dataset into a single column array
xTrainFlat = xTrain.reshape(len(xTrain), (28 * 28))
xTestFlat = xTest.reshape(len(xTest), (28 * 28))

# set up the neural network
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model in 5 iterations
model.fit(xTrainFlat, yTrain, epochs=5)

# evaluate the model
model.evaluate(xTestFlat, yTest)

EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.56.1\bin\gswin64c'

# set up the GUI
root = Tk()
root.title('GUI')
root.geometry('500x600')

canvas = Canvas(root, width=300, height=300, bg="white")
canvas.pack(pady=20)

label = Label(root, text="Prediction: ", font=('Aerial 18'))
label.pack()

label2 = Label(root, text="Degree of accuracy: ", font=('Aerial 10'))
label2.pack()


def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y


def draw_smth(event):
    global lasx, lasy
    canvas.create_oval((lasx, lasy, event.x, event.y),
                       fill='black',
                       width=4)
    lasx, lasy = event.x, event.y


canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)

# clear button
clear_button = ttk.Button(
    root,
    text='Clear!',
    command=lambda: canvas.delete("all")
)

clear_button.pack(
    side=tkinter.RIGHT,
    pady=5,
    padx=50,
    ipadx=5,
    ipady=5,
)


# widget name should be the canvas with drawing
def predict(widget, fileName):
    # save postscript image
    widget.postscript(file=fileName + '.eps')
    # use PIL to convert to PNG
    img = Image.open(fileName + '.eps')
    img.save(fileName + '.png', 'png')
    # convert image to array
    file = r'C:\Users\scott\PycharmProjects\capstone\img.png'
    test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    img_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.bitwise_not(img_resized)

    img_resized = img_resized / 255

    imgFlat = img_resized.reshape(-1, (28 * 28))

    inputPredict = model.predict(imgFlat)

    label["text"] = "Prediction: " + str(np.argmax(inputPredict[0]))

    label2["text"] = "Degree of accuracy: " + str(max(inputPredict[0]))


# predict button
predict_button = ttk.Button(
    root,
    text='Predict!',
    command=lambda: predict(canvas, "img")
)

predict_button.pack(
    side=tkinter.LEFT,
    pady=5,
    padx=50,
    ipadx=5,
    ipady=5,
)


def report():
    # set up and store confusion matrix
    yPredicted = model.predict(xTestFlat)

    yPredictedLabels = [np.argmax(i) for i in yPredicted]

    confusionMatrix = tf.math.confusion_matrix(labels=yTest, predictions=yPredictedLabels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(confusionMatrix, annot=True, fmt='d')
    plt.xlabel('Prediction')
    plt.ylabel('Intended Result')
    plt.show()


# reports button
reports_button = ttk.Button(
    root,
    text='Report',
    command=lambda: report()
)

reports_button.pack(
    side=tkinter.BOTTOM,
    pady=5,
    padx=5,
    ipadx=5,
    ipady=5,
)


def copy():
    root.clipboard_clear()
    root.clipboard_append('shic172@wgu.edu')
    root.update()


# copy button
copy_button = ttk.Button(
    root,
    text='Copy Email',
    command=lambda: copy()
)

copy_button.pack(
    pady=5,
    padx=5,
    ipadx=5,
    ipady=5,
)

root.mainloop()
