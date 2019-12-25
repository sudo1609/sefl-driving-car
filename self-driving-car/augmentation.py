import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator

def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

#shape's của ảnh input
Height, Width, Channels = 66, 200, 3
INPUT_SHAPE = (Height, Width, Channels)

#cắt ảnh
def Crop_image(image):
    return image[60:135, :, :]

#resize lại ảnh về kích thươc chuẩn cho input
def Resize(image):
    return cv2.resize(image, (Width, Height))

#chuyển hệ màu ảnh sang YUV
def RGB_YUV(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

#Pre-process image
def preprocess(image):
    image = Crop_image(image)
    image = Resize(image)
    image = RGB_YUV(image)
    return image

#flip ảnh theo chiều ngược lại
def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle # flip ảnh ngược lại nên phải đổi ngược góc lại lại.
    return image, steering_angle

#dịch chuyển ảnh ngẫu nhiên theo chiều x hoặc y
def random_translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.array([[1. , 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

#Tăng góc lái cho ảnh trái và phải lên 0.2
def choose_image(data_dir, center, left, right, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    else:
        return load_image(data_dir, center), steering_angle

#Thêm giá trị sáng cho ảnh
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #convert sang hệ màu HSV
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) #convert trở lại hệ màu RGB

#Kết hợp tất cả các phương pháp augmentation cho ảnh.
def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_brightness(image)
    return image, steering_angle

#trả lại góc lái tương ứng cho việc training
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, Height, Width, Channels])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers