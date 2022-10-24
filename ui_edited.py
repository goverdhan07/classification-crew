import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import random
import os
import sys
from PIL import Image, ImageTk

window = tk.Tk()

window.title("FOOD CLASSIFICATION AND CALORIE ESTIMATION")

window.geometry("600x560")
window.configure(background ="darkcyan")

title = tk.Label(text="Click below to choose picture for food classification..", background = "darkcyan", fg="Brown", font=("times", 15, "bold italic" ))
##title.grid()
title.grid(column=2, row=0, padx=10, pady=10)
def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'healthyvsunhealthyfood-new-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')
    def stages():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
            hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            #definig the range of red color
            red_lower=np.array([0,0,212],np.uint8)
            red_upper=np.array([131,255,255],np.uint8)
            

            red=cv2.inRange(hsv, red_lower, red_upper)
            kernal = np.ones((5 ,5), "uint8")
            red=cv2.dilate(red, kernal)
            res=cv2.bitwise_and(img, img, mask = red)
            #Tracking the Red Color
            contours,hierarchy =cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            count=0
            
            
            for cnt in contours:
                value = cnt[0,0,0]
                pyval = value.item()
                global sum1
                sum1=0
                sum1=sum1+pyval 
##            print(sum1)
            print("The calorie in the predicted food item is {}%".format(sum1))
##                print(type(cnt[0,0,0]))


    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    #tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 12, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        pred= random.randint(90,98)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]
##        print(model_out)
        print('model {}'.format(np.argmax(model_out)))



        #if model_out > 0.5

        if np.argmax(model_out) == 0:
            stages()
            print("The predicted image of the biriyani is with a accuracy of {} %".format(model_out[0]))
            str_label = 'biriyani'
        elif np.argmax(model_out) == 1:
            stages()
            print("The predicted image of the bisibelebath is with a accuracy of {} %".format(model_out[1]))
            str_label = 'bisibelebath'
        elif np.argmax(model_out) == 2:
            stages()
            print("The predicted image of the butternaan is with a accuracy of {} %".format(model_out[2]))
            str_label = 'butternaan'
        elif np.argmax(model_out) == 3:
            stages()
            print("The predicted image of the chaat is with a accuracy of {} %".format(model_out[3]))
            str_label = 'chaat'
        elif np.argmax(model_out) == 4:
            stages()
            print("The predicted image of the chappati is with a accuracy of {} %".format(model_out[4]))
            str_label = 'chappati'
        elif np.argmax(model_out) == 5:
            stages()
            print("The predicted image of the dhokla is with a accuracy of {} %".format(model_out[5]))
            str_label = 'dhokla'
        elif np.argmax(model_out) == 6:
            stages()
            print("The predicted image of the dosa is with a accuracy of {} %".format(model_out[6]))
            str_label = 'dosa'
        elif np.argmax(model_out) == 7:
            stages()
            print("The predicted image of the idly is with a accuracy of {} %".format(model_out[7]))
            str_label = 'idly'
        elif np.argmax(model_out) == 8:
            stages()
            print("The predicted image of the noodles is with a accuracy of {} %".format(model_out[8]))
            str_label = 'noodles'
        elif np.argmax(model_out) == 9:
            stages()
            print("The predicted image of the upma is with a accuracy of {} %".format(model_out[9]))
            str_label = 'upma'
        elif np.argmax(model_out) == 10:
            stages()
            print("The predicted image of the poori is with a accuracy of {} %".format(model_out[10]))
            str_label = 'poori'
        elif np.argmax(model_out) == 11:
            stages()
            print("The predicted image of the samosa is with a accuracy of {} %".format(model_out[11]))
            str_label = 'samosa'
  
        if str_label == 'biriyani':
            status = "biriyani"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10)          
            predss="Predicted image of  biriyani is  {} % accuracy".format(model_out[0]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)

            
        elif str_label == 'chappati':
            status = "chappati"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  chappati is  {} % accuracy".format(model_out[4]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
##        elif str_label == 'chappati':
##            status = "chappati"
##            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
##            r.grid(column=0, row=5, padx=10, pady=10)
##            predss="Predicted image of  chappati is  {} % accuracy".format(model_out[0]*100)
##            calo="The calorie in the predicted food item is {}%".format(sum1)            
##            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
##            calo.grid(column=0, row=4, padx=10, pady=10)            
##            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
##            calo.grid(column=0, row=5, padx=10, pady=10)            
##            button = tk.Button(text="Exit", command=exit)
##            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'bisibelebath':
            status="bisibelebath"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  bisibelebath is  {} % accuracy".format(model_out[1]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'butternaan':
            status = "butternaan"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  butternaan is  {} % accuracy".format(model_out[2]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'chaat':
            status = "chaat"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  chaat is  {} % accuracy".format(model_out[3]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'dhokla':
            status = "dhokla"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  dhokla is  {} % accuracy".format(model_out[5]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'dosa':
            status = "dosa"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  dosa is  {} % accuracy".format(model_out[6]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'idly':
            status = "idly"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  idly is  {} % accuracy".format(model_out[7]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'noodles':
            status = "noodles"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  noodles is  {} % accuracy".format(model_out[8]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'upma':
            status = "upma"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  upma is  {} % accuracy".format(model_out[9]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'poori':
            status = "poori"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  poori is  {} % accuracy".format(model_out[10]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'samosa':
            status = "samosa"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="darkcyan", fg="Brown", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  samosa is  {} % accuracy".format(model_out[11]*100)
            calo="The calorie in the predicted food item is {}%".format(sum1)            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
       

def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='C:\\Users\\HP\\Downloads\\food_classification\\food_classification\\test', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    dst = "testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    if os.path.split(fileName)[-1].split('.') == 'h (1)':
        print('dfdffffffffffffff')
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="225", width="450")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    global button2
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)
button1 = tk.Button(text="Get Photo", command = openphoto)
button1.grid(column=2, row=1, padx=5, pady = 5)



window.mainloop()



