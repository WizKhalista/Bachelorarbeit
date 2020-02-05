import numpy as np
from keras.models import load_model
import Leap.Leap, sys
from Leap.Leap import Leap


class Leaplistener(Leap.Listener):
    def on_connect(self, controller):
        print("Leap Motion Connected")

    def on_disconnect(self, controller):
        print("Leap Motion Disconnected")

    def on_init(self, controller):
        print("Initialized")

    def on_images(self, controller):
       # print("Images available")
        leapImage = controller.images[0]
        np_img = np.array(leapImage)
        np_img = np.resize(np_img, (1,64,64,1))
        print(np_img.shape)
        #image_buffer_pointer= leapImage.data_pointer
        #ctype_array_def = ctypes.c_ubyte * leapImage.height * leapImage.width
        #as_numpy_array = np.ctypeslib.as_array(ctype_array_def.from_address(int(image_buffer_pointer)))
        #numpy_array = np.array(as_numpy_array)
        #numpy_array = numpy_array.resize(numpy_array(1,64,64,1))

        #print(type(numpy_array))
        #print(numpy_array.shape)
        classes = model.predict_classes(np_img)
        print(classes)


model = load_model('FingersR.h5')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



#def addToNpArray(npArray):
#    numpy_array.append(npArray)




def main():

    controller = Leap.Controller()
    listener = Leaplistener()
    controller.add_listener(listener)
    controller.set_policy(controller.POLICY_IMAGES)

    print("press enter to quit")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt():
        pass
    finally:

        controller.remove_listener(listener)


if __name__ == "__main__":
    main()