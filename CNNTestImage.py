from keras.models import load_model
from keras.backend import set_session
import numpy as np
import ctypes
from Leap import Leap
import sys
from PIL import Image as pil_image
from matplotlib import pyplot as plt
import tensorflow
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keyboard
from sklearn.metrics import confusion_matrix, classification_report
from tkinter import *
from tkinter import ttk
import time



#Variables
list_of_images = []
global i
i=0

#Set session
sess=tensorflow.Session()
set_session(sess)

global model
model = load_model('Trained Models/vgg_newset.h5')
#plot_model(model, to_file='InceptionCNN.png',show_layer_names=True,show_shapes=True)
model.summary()
#Define the graph
global graph
graph = tensorflow.compat.v1.get_default_graph()

controller = Leap.Controller()
controller.set_policy(controller.POLICY_IMAGES)



def evaluate_generator():
    print('evaluating')
    test_datagen = ImageDataGenerator(brightness_range=(0.2, 1.4))
    test_gen = test_datagen.flow_from_directory('E:/Datensatz/test',target_size=(240, 300), class_mode='categorical', batch_size=1)
    evaluated = model.evaluate(test_gen)
    print(model.metrics_names)
    print('ergebnis:',evaluated)

def create_confusion_matrix():
    print('creating confusion matrix')
    test_datagen = ImageDataGenerator(brightness_range=(0.2, 1.4))
    test_gen = test_datagen.flow_from_directory('E:/Datensatz/test',target_size=(240, 300),class_mode='categorical', batch_size=1, shuffle=False)
    predicted = model.predict_generator(test_gen)
    y_pred = model.predict_generator(test_gen, 68)
    Y_pred = np.argmax(y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_gen.classes, Y_pred))

#for evaluating or testing a Set
#evaluate_generator()
#create_confusion_matrix()
def predict_img():

    images = controller.images
    image0 = images[0]
    image1 = images[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy arrays
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))
    #concat 2 arrays to 1
    arrays = np.concatenate((as_numpy_array0,as_numpy_array1),axis=1)
    #save as image to rescale and predict
    global i
    img_save = pil_image.fromarray(arrays)
    img_save.save('test.jpg','jpeg')
    img = image.load_img('test.jpg',target_size=(240,300), color_mode='rgb')
    #Show image before predicting
    #plt.imshow(img)
    #plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    predicted = model.predict(x)
    class_pred = (predicted.argmax(axis=-1))
    max_value = predicted.max()
    print(max_value)
    #print(pred)
    print(class_pred)
    interpret_label(class_pred)
    return interpret_label(class_pred), max_value

def interpret_label(label):
    if label ==[0]:
        print("button1")
        return "button1"
    if label ==[1]:
        print("button2")
        return "button2"
    if label ==[2]:
        print("button3")
        return "button3"
    if label ==[3]:
        print("button4")
        return "button4"
    if label ==[4]:
        print("fist")
        return "fist"
    if label ==[5]:
        print("flach")
        return"flach"
    if label ==[6]:
        print("gespreizt")
        return "gespreizt"
    if label ==[7]:
        print("ok")
        return"ok"
    if label ==[8]:
        print("peace")
        return "peace"
    if label ==[9]:
        print("pencil")
        return "pencil"
    if label ==[10]:
        print("pistol")
        return "pistol"

while False:
    if keyboard.is_pressed('p'):
        time.sleep(2)
        predict_img()


#Save Images yes or no?
global bool
bool = False

#
#Leap Motion Things
class LeapListener(Leap.Listener):
    def on_connect(self, controller):
        print("Connected")

    def on_images(self, controller):
        predict_img()
        #imageList = controller.images
        #predict_img()
        #global bool
        #if bool==True:
        #    save_images(imageList)









def save_images(imageList):
    global bool
    bool=False
    i = len(list_of_images)
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image0.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    #Save images

    img_saver1 = Image.fromarray(as_numpy_array0)
    img_saver1.save('C:/Users/Marcel/Pictures/BA/LeapCameraImages/testfile_%i.jpg'%i,'jpeg')
    i += 1
    img_saver2 = Image.fromarray((as_numpy_array1))
    img_saver2.save('C:/Users/Marcel/Pictures/BA/LeapCameraImages/testfile_%i.jpg'%i,'jpeg')
    i += 1


def image_gen(imageList):
    i = len(list_of_images)
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image0.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    #concatting arrays to one, now got both images in one numpy array

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1),axis=1)

    print(arrays.shape)
    #Show numpy array

    plt.imshow(arrays)
    plt.show()

    #grayscale image over RGB channels repeated
    arrays = np.repeat(arrays[...,np.newaxis],3,-1)
    print('after repeating over channels:',arrays.shape)
    #expand dimensions to fit Input
    arrays = np.expand_dims(arrays, axis=0)
    print('after adding axis=0:',arrays.shape)

    #predict with CNN
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        predictions = model.predict(arrays)
        print('Predictions', predictions)




"""def main():
    controller = Leap.Controller()
    listener = LeapListener()
    controller.add_listener(listener)
    controller.set_policy(controller.POLICY_IMAGES)


    print("press enter to quit")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt():
        pass
    finally:

        controller.remove_listener(listener)

#if __name__ == "__main__":
#    main()"""



#Button Functions
predict = True


def start():
    global predict
    predict = True

def stop():
    global predict
    predict = False

#Creating GUI
def main():
    Root = Tk()
    Root.minsize(500,300)
    Root.title("Predict")


    def predicting(m1):
        if predict:
            time.sleep(3)
            prediction, value = predict_img()
            m1.config(text=prediction)
            #m2.config(text=value)
            Root.update()
            while True:
                time.sleep(1)
                prediction = predict_img()
                m1.config(text=prediction)
                #m2.config(text=value)
                Root.update()



    label = ttk.Label(Root, text='Predicted', background="green")
    label.place(x=70,y=100)
    MyLabel1 = ttk.Label(Root, text = 'The button has not been pressed.')
    MyLabel1.place(x=130,y=100)
    #MyLabel2 = ttk.Label(Root, text='button not yet pressed')
    #MyLabel2.place(x=130, y=130)

    #Erstellen der Legende
    #Beschriftung der Posen
    l1 = ttk.Label(Root, text = 'Legende')
    l1.place(x=350,y=20)
    l2 = ttk.Label(Root, text='button1', background="white")
    l2.place(x=400,y=40)
    l3 = ttk.Label(Root, text='button2', background="white")
    l3.place(x=400,y=60)
    l4 = ttk.Label(Root, text='button3', background='white')
    l4.place(x=400,y=80)
    l5 = ttk.Label(Root, text='button4', background="white")
    l5.place(x=400,y=100)
    l6 = ttk.Label(Root, text='fist', background="white")
    l6.place(x=400, y=120)
    l7 = ttk.Label(Root, text='flach', background="white")
    l7.place(x=400, y=140)
    l8 = ttk.Label(Root, text='gespreizt', background="white")
    l8.place(x=400, y=160)
    l9 = ttk.Label(Root, text='ok', background="white")
    l9.place(x=400, y=180)
    l10 = ttk.Label(Root, text='peace', background="white")
    l10.place(x=400, y=200)
    l11 = ttk.Label(Root, text='pencil', background="white")
    l11.place(x=400, y=220)
    l12 = ttk.Label(Root, text='pistol', background="white")
    l12.place(x=400, y=240)

    #Beschriftung der Labels als Zahl
    x2 = ttk.Label(Root, text='[0]', background="white")
    x2.place(x=350, y=40)
    x3 = ttk.Label(Root, text='[1]', background="white")
    x3.place(x=350, y=60)
    x4 = ttk.Label(Root, text='[2]', background='white')
    x4.place(x=350, y=80)
    x5 = ttk.Label(Root, text='[3]', background="white")
    x5.place(x=350, y=100)
    x6 = ttk.Label(Root, text='[4]', background="white")
    x6.place(x=350, y=120)
    x7 = ttk.Label(Root, text='[5]', background="white")
    x7.place(x=350, y=140)
    x8 = ttk.Label(Root, text='[6]', background="white")
    x8.place(x=350, y=160)
    x9 = ttk.Label(Root, text='[7]', background="white")
    x9.place(x=350, y=180)
    x10 = ttk.Label(Root, text='[8]', background="white")
    x10.place(x=350, y=200)
    x11 = ttk.Label(Root, text='[9]', background="white")
    x11.place(x=350, y=220)
    x12 = ttk.Label(Root, text='[10]', background="white")
    x12.place(x=350, y=240)

    #Button zum Predicten
    MyButton1 = ttk.Button(Root, text = 'Start predicting', command = lambda: predicting(MyLabel1))
    #MyButton1.config(height=50,width=50)
    MyButton1.place(x=100,y=50)
    Root.mainloop()

if __name__ == "__main__":
    main()


