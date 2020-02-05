import sys
from Leap import Leap


class LeapListener(Leap.Listener):

    def on_init(self, controller):
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")

    def on_disconnect(self, controller):
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_images(self, controller):
        images = controller.images
        left_image = images[0]
        right_image = images[1]
        #data = (numpy.frombuffer(left_image.data))
        #npDataArray.append(numpy.frombuffer(right_image.data))
        print(left_image.data)





#Images are 240x640 (height x width)
imageList = []
#Image Data stored in this Array
imageDataArray = []
#Image Data stored as NumpyArray
npDataArray=[]

def addToImageList(images):
    imageList.append(images)
    print("images added to imagelist" + "imagelist now contains" + str(len(imageList)) + "images" )



#def getData(imageList):
#    for img in imageList:
#        imageDataArray.append(img.data)

#def toNpDataArray(imageDataArray):
#    for data in imageDataArray:
#        npData = numpy.frombuffer(data,count=-1,offset=0)
#        npDataArray.append(npData)


def main():
    listener = LeapListener()
    controller = Leap.Controller()
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
