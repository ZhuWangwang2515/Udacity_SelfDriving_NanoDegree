# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:45:19 2018

@author: 13120
"""
import csv
import os #os提供很多方法来处理文件和目录
import cv2
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Convolution2D,Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
import keras.backend.tensorflow_backend as KTF
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
######################################
# TODO: set the gpu memory using fraction #
#####################################
def get_session(gpu_fraction=0.3):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session(0.6))  # using 40% of total GPU Memory



lines = []
# ../表示当前文件所在的目录的上一级目录
# ./表示当前文件所在的目录（可以省略）
# /表示当前站点的根目录
with open('./data/driving_log.csv')  as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#train_set,validation_set = train_test_split(lines,test_size = 0.4)
#        
#def generator(input_set,batch_size = 32):
#    num_set = len(input_set)
#    while 1:
#        shuffle(input_set)
#        for offset in range(0,num_set,batch_size):
#            batch_set = input_set[offset:offset+batch_size]
#            images = []
#            steering_angles = []
#            for batch_image in batch_set:
#                source_path = batch_image[0]
#                filename = source_path.split('\\')[-1]
#                current_path = './Data/IMG/'+filename #将读取的绝对路径转换为相对路径
#                image = cv2.imread(current_path)
#                # print(filename)
#                # print(image)
#                images.append(image)
#                image_filp = cv2.flip(image,1)
#                images.append(image_filp)
#                steering_angle = float(batch_image[3])
#                steering_angles.append(steering_angle)
#                steering_angle_flip = steering_angle * -1.0
#                steering_angles.append(steering_angle_flip)  
#                
#                correction = 0.2
#                
#                # 取出左侧图片
#                left_path = batch_image[1]
#                left_filename = left_path.split('\\')[-1]
#                left_picture_path =  './Data/IMG/'+left_filename
#                left_image = cv2.imread(left_picture_path)
#                images.append(left_image)
#                # 偏移左侧图片对应的方向盘转角
#                steering_center = float(batch_image[3])
#                steering_left = steering_center + correction
#                steering_angles.append(steering_left)
#                # 取出右侧图片
#                right_path = batch_image[2]
#                right_filename = right_path.split('\\')[-1]
#                right_picture_path = './Data/IMG/' + right_filename
#                right_image = cv2.imread(right_picture_path)
#                images.append(right_image)
#                # 偏移右侧图片对应的方向盘转角
#                steering_right = steering_center - correction
#                steering_angles.append(steering_right)
#            X_train = np.array(images)
#            y_train = np.array(steering_angles)
#            yield shuffle(X_train,y_train)
        
images = []
measurments = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './Data/IMG/'+filename #将读取的绝对路径转换为相对路径
    image = cv2.imread(current_path)
    # print(filename)
    # print(image)
    images.append(image)
    image_filp = cv2.flip(image,1)
    #images.append(image_filp)
    measurment = float(line[3])
    measurments.append(measurment)
    measurment_flip = measurment * -1.0
    #measurments.append(measurment_flip)  
    
    correction = 0.2
    
    # 取出左侧图片
    left_path = line[1]
    left_filename = left_path.split('\\')[-1]
    left_picture_path =  './Data/IMG/'+left_filename
    left_image = cv2.imread(left_picture_path)
    images.append(left_image)
    # 偏移左侧图片对应的方向盘转角
    steering_center = float(line[3])
    steering_left = steering_center + correction
    measurments.append(steering_left)
    # 取出右侧图片
    right_path = line[2]
    right_filename = right_path.split('\\')[-1]
    right_picture_path = './Data/IMG/' + right_filename
    right_image = cv2.imread(right_picture_path)
    images.append(right_image)
    # 偏移右侧图片对应的方向盘转角
    steering_right = steering_center - correction
    measurments.append(steering_right)
    
    
X_train = np.array(images)
y_train = np.array(measurments)
# print(X_train[10])

# 根据Keras建立神经网络
model = Sequential() 
model.add(Lambda(lambda x:x / 255.0 -0.5 , input_shape = (160,320,3))) 
model.add(Cropping2D(cropping=((76,25),(0,0))))
model.add(Conv2D(24,(5,5),activation='relu',padding='SAME'))#
model.add(MaxPooling2D())
#model.add(Dropout(0.2))
model.add(Conv2D(36,(5,5),activation='relu',padding='SAME'))#
model.add(MaxPooling2D())
#model.add(Dropout(0.2))
model.add(Conv2D(48,(5,5),activation='relu',padding='SAME'))#
model.add(MaxPooling2D())
model.add(Conv2D(64,(5,5),activation='relu',padding='SAME'))
#model.add(MaxPooling2D())
model.add(Conv2D(64,(5,5),activation='relu'))
#model.add(MaxPooling2D())
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(120))
#model.add(Dropout(0.5))
model.add(Dense(84))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer = 'adam')
print('目前为止，测试通过')

#train_generator = generator(train_set,batch_size = 64)
#validation_generator = generator(validation_set,batch_size = 64)
#history_object = model.fit_generator(train_generator, steps_per_epoch =len(train_set), 
#                                     validation_data = validation_generator,validation_steps = len(validation_set), 
#                                     nb_epoch=3, verbose=1)
history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=2)
model.save('C:\\Users\\13120\\Desktop\\优达学城\\Homework\\CarND-Behavioral-Cloning-P3-master\\model.h5')
# model.save('D:\\研三上\\张老师\\优达学城\\Homework\\CarND-Behavioral-Cloning-P3-master\\model.h5')
print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc = 'upper right')
plt.show()
 










