import keyboard, ctypes, numpy as np
from Leap import Leap
from PIL import Image
import time
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image
#from keras.models import load_model
#from keras.preprocessing.image import img_to_array


global i
i = 12
controller = Leap.Controller()
controller.set_policy(controller.POLICY_IMAGES)

#methods for saving Images via Keypress, saving into E:/BA/Datensatz/train oder valid
#Save button1 img (1)
def save_images1(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/button1/button1%i.jpg' % i, 'jpeg')
    i += 1
    print('button1 Image Saved!')
#Save button2 img (2)
def save_images2(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/button2/button2%i.jpg' % i, 'jpeg')
    i += 1
    print('button2 Image Saved!')
#Save button3 img    (3)
def save_images3(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/button3/button3%i.jpg' % i, 'jpeg')
    i += 1
    print('button3 Image Saved!')
#save fist img   (4)
def save_images4(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/fist/fist%i.jpg' % i, 'jpeg')
    i += 1
    print('fist Image Saved!')
#save flach img (5)
def save_images5(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/flach/flach%i.jpg' % i, 'jpeg')
    i += 1
    print('flach Image Saved!')
#save gespreitzt img (6)
def save_images6(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/gespreizt/gespreizt%i.jpg' % i, 'jpeg')
    i += 1
    print('gespreizt Image Saved!')
#save ok img (7)
def save_images7(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/ok/ok%i.jpg' % i, 'jpeg')
    i += 1
    print('ok Image Saved!')
#save peace img   (8)
def save_images8(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/peace/peace%i.jpg' % i, 'jpeg')
    i += 1
    print('peace Image Saved!')
#save pencil img (9)
def save_images9(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/pencil/pencil%i.jpg' % i, 'jpeg')
    i += 1
    print('pencil Image Saved!')
#pistol img (0)
def save_images0(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0,as_numpy_array1), axis=1)
    #Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/pistol/pistol%i.jpg'%i, 'jpeg')
    i += 1
    print('pistol Image Saved!')


#save snake pose img (w)
def save_imagesw(imageList):
    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    arrays = np.concatenate((as_numpy_array0, as_numpy_array1), axis=1)
    # Save image
    global i
    img_saver1 = Image.fromarray(arrays)
    img_saver1.save('newDataset/test/button4/button4%i.jpg' % i, 'jpeg')
    i += 1
    print('button4 Image Saved!')
#save tiger pose img (e)
def save_imagese(imageList):

    image0 = imageList[0]
    image1 = imageList[1]
    image0_buffer_ptr = image0.data_pointer
    image1_buffer_ptr = image1.data_pointer
    ctype_array_def = ctypes.c_ubyte * image0.width * image1.height

    # as numpy array
    as_numpy_array0 = ctype_array_def.from_address(int(image0_buffer_ptr))
    as_numpy_array0 = np.ctypeslib.as_array(as_numpy_array0)

    as_numpy_array1 = ctype_array_def.from_address((int(image1_buffer_ptr)))
    as_numpy_array1 = np.ctypeslib.as_array((as_numpy_array1))

    #Save images
    global i
    img_saver1 = Image.fromarray(as_numpy_array0)
    #img_saver1.save('C:/Users/Marcel/Pictures/BA/Dataset/1/1%i.jpg'%i,'jpeg')
    #i += 1
    img_saver2 = Image.fromarray((as_numpy_array1))
    #img_saver2.save('C:/Users/Marcel/Pictures/BA/Dataset/1/1%i.jpg'%i,'jpeg')
    #i += 1
    new_img = Image.new('RGB',(image0.width+image1.width,image0.height))
    new_img.paste(img_saver1, (0,0))
    new_img.paste(img_saver2, (image1.width,0))
    new_img.save('newDataset/tiger_pose/tiger_pose%i.jpg'%i,'jpeg')
    i += 1
    print('tiger pose Image Saved!')
#True for Dataset Creation
bool = True
while bool:
    if keyboard.is_pressed('1'):
            time.sleep(3)
            imageList = controller.images
            save_images1(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images1(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images1(imageList)
    if keyboard.is_pressed('2'):
                time.sleep(3)
                imageList = controller.images
                save_images2(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images2(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images2(imageList)
    if keyboard.is_pressed('3'):
            time.sleep(3)
            imageList = controller.images
            save_images3(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images3(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images3(imageList)
    if keyboard.is_pressed('w'):
            time.sleep(3)
            imageList = controller.images
            save_imagesw(imageList)
            time.sleep(2)
            imageList = controller.images
            save_imagesw(imageList)
            time.sleep(2)
            imageList = controller.images
            save_imagesw(imageList)
    if keyboard.is_pressed('4'):
                time.sleep(3)
                imageList = controller.images
                save_images4(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images4(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images4(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images4(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images4(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images4(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images4(imageList)
    if keyboard.is_pressed('5'):
            time.sleep(3)
            imageList = controller.images
            save_images5(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images5(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images5(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images5(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images5(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images5(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images5(imageList)

    if keyboard.is_pressed('6'):
                time.sleep(3)
                imageList = controller.images
                save_images6(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images6(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images6(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images6(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images6(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images6(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images6(imageList)
    if keyboard.is_pressed('7'):
            time.sleep(3)
            imageList = controller.images
            save_images7(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images7(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images7(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images7(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images7(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images7(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images7(imageList)
    if keyboard.is_pressed('8'):
                time.sleep(3)
                imageList = controller.images
                save_images8(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images8(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images8(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images8(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images8(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images8(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images8(imageList)
    if keyboard.is_pressed('9'):
            time.sleep(3)
            imageList = controller.images
            save_images9(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images9(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images9(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images9(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images9(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images9(imageList)
            time.sleep(2)
            imageList = controller.images
            save_images9(imageList)
    if keyboard.is_pressed('0'):
                time.sleep(3)
                imageList = controller.images
                save_images0(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images0(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images0(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images0(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images0(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images0(imageList)
                time.sleep(2)
                imageList = controller.images
                save_images0(imageList)


"""#train_datagen = ImageDataGenerator(,shear_range=0.15, vertical_flip=False, brightness_range=(0.1,0.9))
test_datagen = ImageDataGenerator(brightness_range=(0.2,1.0))
#valid_datagen = ImageDataGenerator(height_shift_range=0.22,shear_range=0.15, vertical_flip=False, horizontal_flip=True, brightness_range=(0.1,0.9))
#train_generator = train_datagen.flow_from_directory('E:/BA/Dataset/train', target_size=(480,1280),save_to_dir='E:/BA/train-generated', class_mode='categorical', batch_size=132)
#valid_generator = valid_datagen.flow_from_directory('E:/BA/Dataset/valid', target_size=(480,1280), save_to_dir='E:/BA/valid-generated', class_mode = 'categorical', batch_size=48)
test_generator = test_datagen.flow_from_directory('E:/BA/Datensatz/test',save_to_dir='E:/BA/Datensatz/test_generated', target_size = (240,1280), batch_size=5, save_format='jpg')





i=0
for batch in test_generator:
    i += 1
    if(i>=10):
        break"""






