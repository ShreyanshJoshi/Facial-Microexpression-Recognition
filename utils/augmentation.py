import tensorflow as tf
import numpy as np
import random
from keras.preprocessing.image import load_img, img_to_array
from utils.common import Augmentation
from utils.miscellaneous import get_frames_3dcnn

dir = "C:/Users/Shreyansh/Desktop/Microexpression Detection/SAMM/"

'''A helper function, used in the case when all 9 frames of a datapoint are stacked depthwise. It augments each of the 9 frames in a similar manner (since they 
would be stacked one behind the other), by randomly generating parameters for augmentation, and randomly deciding which type of augmentation has to be performed. 
It then stacks the augmented frames, one behind the other (depthwise) and returns it. '''
def augment_helper_3dimages_stacked(frames):
    img_arr1 = frames[0]
    img_arr1_1 = frames[1]
    img_arr1_2 = frames[2]
    img_arr1_3 = frames[3]
    img_arr2 = frames[4]
    img_arr2_1 = frames[5]
    img_arr2_2 = frames[6]
    img_arr2_3 = frames[7]
    img_arr3 = frames[8]

    directions = ['right', 'left']
    dir_index = random.randint(0,1)
    shift = random.randint(1,15)
    angle = [random.uniform(1,13), random.uniform(348,359)]
    angle_index = random.randint(0,1)
    mean = random.uniform(1,6)
    sigma = random.uniform(1,10)
    flag = 0
    
    while flag==0:
        coin = random.randint(0,1)        # Translation
        if coin==1:
            flag = 1
            img_arr1 = Augmentation.translate(img_arr1, shift=shift, direction=directions[dir_index])
            img_arr1_1 = Augmentation.translate(img_arr1_1, shift=shift, direction=directions[dir_index])
            img_arr1_2 = Augmentation.translate(img_arr1_2, shift=shift, direction=directions[dir_index])
            img_arr1_3 = Augmentation.translate(img_arr1_3, shift=shift, direction=directions[dir_index])
            img_arr2 = Augmentation.translate(img_arr2, shift=shift, direction=directions[dir_index])
            img_arr2_1 = Augmentation.translate(img_arr2_1, shift=shift, direction=directions[dir_index])
            img_arr2_2 = Augmentation.translate(img_arr2_2, shift=shift, direction=directions[dir_index])
            img_arr2_3 =Augmentation.translate(img_arr2_3, shift=shift, direction=directions[dir_index])
            img_arr3 = Augmentation.translate(img_arr3, shift=shift, direction=directions[dir_index])

        coin = random.randint(0,1)        # Rotation
        if coin==1:
            flag = 1  
            img_arr1 = Augmentation.rotate_img(img_arr1, angle[angle_index])
            img_arr1_1 = Augmentation.rotate_img(img_arr1_1, angle[angle_index])
            img_arr1_2 = Augmentation.rotate_img(img_arr1_2, angle[angle_index])
            img_arr1_3 = Augmentation.rotate_img(img_arr1_3, angle[angle_index])
            img_arr2 = Augmentation.rotate_img(img_arr2, angle[angle_index])
            img_arr2_1 = Augmentation.rotate_img(img_arr2_1, angle[angle_index])
            img_arr2_2 = Augmentation.rotate_img(img_arr2_2, angle[angle_index])
            img_arr2_3 = Augmentation.rotate_img(img_arr2_3, angle[angle_index])
            img_arr3 = Augmentation.rotate_img(img_arr3, angle[angle_index])

        coin = random.randint(0,1)        # Gaussian noise
        if coin==1:
            flag = 1    
            img_arr1 = Augmentation.gaussian_noise(img_arr1, mean=mean, sigma=sigma)
            img_arr1_1 = Augmentation.gaussian_noise(img_arr1_1, mean=mean, sigma=sigma)
            img_arr1_2 = Augmentation.gaussian_noise(img_arr1_2, mean=mean, sigma=sigma)
            img_arr1_3 = Augmentation.gaussian_noise(img_arr1_3, mean=mean, sigma=sigma)
            img_arr2 = Augmentation.gaussian_noise(img_arr2, mean=mean, sigma=sigma)
            img_arr2_1 = Augmentation.gaussian_noise(img_arr2_1, mean=mean, sigma=sigma)
            img_arr2_2 = Augmentation.gaussian_noise(img_arr2_2, mean=mean, sigma=sigma)
            img_arr2_3 = Augmentation.gaussian_noise(img_arr2_3, mean=mean, sigma=sigma)
            img_arr3 = Augmentation.gaussian_noise(img_arr3, mean=mean, sigma=sigma)
      
    final = np.concatenate((img_arr1, img_arr1_1, img_arr1_2, img_arr1_3, img_arr2, img_arr2_1, 
                                img_arr2_2, img_arr2_3, img_arr3), axis=-1)
    return final


'''Augmenting images in the case when all 9 frames of a datapoint are stacked depthwise. In each class, augmentation is done after taking into consideration number 
of images originally present in that class. For instance, in case of multiclass classification, since 'anger' has the most number of images originally, no 
augmentation was done. Also, before augmenting, the original image (it's 9 frames) has also been added. It's also important to note that adding a huge number of 
augmented images, just to increase the dataset size (or to balance out distribution) is a bad idea as it introduces redundancy into the system.'''
def augment_3dimages_stacked(classes, type):
    data = []                                           # Contains list of augmented images
    labels = []                                         # Contains labels of images after augmentation

    if type=="multiclass":        
        for i in classes[0]:
            frames = get_frames_3dcnn(i)
            final = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], 
                                    frames[6], frames[7], frames[8]), axis=-1)
            data.append(final)
            labels.append(0)

            for j in range(1):
                final = augment_helper_3dimages_stacked(frames)
                data.append(final)
                labels.append(0)


        for i in classes[1]:
            frames = get_frames_3dcnn(i)
            final = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], 
                                    frames[6], frames[7], frames[8]), axis=-1)
            data.append(final)
            labels.append(1)

            for j in range(3):
                final = augment_helper_3dimages_stacked(frames)
                data.append(final)
                labels.append(1)


        for i in classes[2]:
            frames = get_frames_3dcnn(i)
            final = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], 
                                    frames[6], frames[7], frames[8]), axis=-1)
            data.append(final)
            labels.append(2)


        for i in classes[3]:
            frames = get_frames_3dcnn(i)
            final = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], 
                                    frames[6], frames[7], frames[8]), axis=-1)
            data.append(final)
            labels.append(3)

            for j in range(5):
                final = augment_helper_3dimages_stacked(frames)
                data.append(final)
                labels.append(3)


        for i in classes[4]:
            frames = get_frames_3dcnn(i)
            final = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], 
                                    frames[6], frames[7], frames[8]), axis=-1)
            data.append(final)
            labels.append(4)

            for j in range(5):
                final = augment_helper_3dimages_stacked(frames)
                data.append(final)
                labels.append(4)


        for i in classes[5]:
            frames = get_frames_3dcnn(i)
            final = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], 
                                    frames[6], frames[7], frames[8]), axis=-1)
            data.append(final)
            labels.append(5)

            for j in range(4):
                final = augment_helper_3dimages_stacked(frames)
                data.append(final)
                labels.append(5)


        for i in classes[6]:
            frames = get_frames_3dcnn(i)
            final = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], 
                                    frames[6], frames[7], frames[8]), axis=-1)
            data.append(final)
            labels.append(6)

            for j in range(7):
                final = augment_helper_3dimages_stacked(frames)
                data.append(final)
                labels.append(6)

        return [data, labels]
    
    elif type=="binary":
        for i in classes[0]:
            frames = get_frames_3dcnn(i)

            final = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], 
                                    frames[6], frames[7], frames[8]), axis=-1)
            data.append(final)
            labels.append(0)

            for j in range(2):
                final = augment_helper_3dimages_stacked(frames)
                data.append(final)
                labels.append(0)

        for i in classes[1]:                              
            frames = get_frames_3dcnn(i)
            final = np.concatenate((frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], 
                                    frames[6], frames[7], frames[8]), axis=-1)
            data.append(final)
            labels.append(1)

        return [data, labels]
    
    else:
        raise ValueError("Invalid argument")


'''A helper function, used in the case when all 9 frames are considered as individual datapoints during training. This function randomly generates parameters for 
augmentation, and decides randomly which type of augmentation has to be performed, and finally returns the augmented image (array). '''
def augment_helper_2dimages_open(img_arr):
    directions = ['right', 'left']
    dir_index = random.randint(0,1)
    shift = random.randint(1,17)
    angle = [random.uniform(1,15), random.uniform(345,359)]
    angle_index = random.randint(0,1)
    mean = random.uniform(1,7)
    sigma = random.uniform(1,12)
    flag = 0
    
    while flag==0:
        coin = random.randint(0,1)        # Translation
        if coin==1:
            flag = 1
            img_arr = Augmentation.translate(img_arr, shift=shift, direction=directions[dir_index])

        coin = random.randint(0,1)        # Rotation
        if coin==1:
            flag = 1  
            img_arr = Augmentation.rotate_img(img_arr, angle[angle_index])

        coin = random.randint(0,1)        # Gaussian noise
        if coin==1:
            flag = 1   
            img_arr = Augmentation.gaussian_noise(img_arr, mean=mean, sigma=sigma) 
      
    return img_arr


'''Another helper function, used in the case when all 9 frames are considered as individual datapoints during training. This function actually calls the function 
that does the augmenting (another helper function), and stored the labels and augmented image into their corresponding lists. '''
def augment_add_to_list_2dimages_open(train_labels, train_data, j, i):
    train_labels.append(i[0])
    img_arr = augment_helper_2dimages_open(img_to_array(load_img(dir + j, color_mode="grayscale", target_size=(128,128))))
    train_data.append(img_arr)


'''From the earlier returned list of classes (with labels and the paths of corresponding 9 frames), that was divided into training and testing sets, the below 
function, seperates labels from the actual data, for the training set, by appending both in seperate lists and returning them. Each of the 9 frames for a datapoint
in the train dataset, is considered as a single datapoint for this task (majority voting policy). Augmentation is performed (or not) depending on the parameters 
passed. 
In case augmentation is performed, it is done after taking into consideration number of images originally present in that class. For instance, in case of multiclass 
classification, since anger has the most number of images originally, no augmentation was done. Also, before augmenting, the original image has also been added. ''' 
def augment_2dimages_open(train_list, type, augment=False):
    
    train_data = []
    train_labels = []

    if augment==False:
        for i in train_list:        
            for j in i[1]:
                train_labels.append(i[0])
                train_data.append(np.array(img_to_array(load_img(dir + j, color_mode="grayscale", target_size=(128,128)))))
        
        return [train_data, train_labels]
    
    else:
        for i in train_list:        
            for j in i[1]:
                train_labels.append(i[0])
                train_data.append(np.array(img_to_array(load_img(dir + j, color_mode="grayscale", target_size=(128,128)))))
                
                if type=="multiclass":
                    if i[0]==2:                             # No need to augment anger as it has most images
                        continue

                    coin = random.randint(0,1)
                    if coin==1:                             # Whether or not to augment is decided by the flip of a coin
                        if i[0]==6:
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)

                        elif i[0]>=3:
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)

                        elif i[0]==1:
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)

                        elif i[0]==0:
                            augment_add_to_list_2dimages_open(train_labels, train_data, j, i)
                
                else:
                    if i[0]==1:                            # No need to augment negative images.
                        continue

                    coin = random.randint(0,1)
                    if coin==1:
                        augment_add_to_list_2dimages_open(train_labels, train_data, j, i)
                        augment_add_to_list_2dimages_open(train_labels, train_data, j, i)

        return [train_data, train_labels]