import os
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import tensorflow as tf
import matplotlib.pyplot as plt

dir = "C:/Users/Shreyansh/Desktop/Microexpression Detection/SAMM/"

# Returns the correct subject number (as a string)
def get_subject(sub):                                       
    if sub < 10:
        return '00' + str(sub)
    
    else:
        return '0' + str(sub)


 # Returns the correct onset frame number (as a string)
def get_on_frame(df, new_sub, filename, onframe):          
    if onframe < 1000:
        if os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_0' + str(onframe) + '.jpg') ==True:
            return "0" + str(onframe)
        
        elif os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_00' + str(onframe) + '.jpg') ==True:
            return "00" + str(onframe)
    
    else:
        if os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_' + str(onframe) + '.jpg') ==True:
            return str(onframe)

        elif os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_0' + str(onframe) + '.jpg') ==True:
            return "0" + str(onframe) 

        else:
            return -1


# Returns the correct apex frame number (as a string)
def get_apex_frame(df, new_sub, filename, apexframe):       
    if apexframe < 1000:
        if os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_0' + str(apexframe) + '.jpg') ==True:
            return "0" + str(apexframe)
        
        elif os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_00' + str(apexframe) + '.jpg') ==True:
            return "00" + str(apexframe)
    
    else:
        if os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_' + str(apexframe) + '.jpg') ==True:
            return str(apexframe)

        elif os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_0' + str(apexframe) + '.jpg') ==True:
            return "0" + str(apexframe)
        
        else:
            return -1


# Returns the correct offset frame number (as a string)
def get_off_frame(df, new_sub, filename, offframe):       
    if offframe < 1000:
        if os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_0' + str(offframe) + '.jpg') ==True:
            return "0" + str(offframe)
        
        elif os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_00' + str(offframe) + '.jpg') ==True:
            return "00" + str(offframe)
    
    else:
        if os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_' + str(offframe) + '.jpg') ==True:
            return str(offframe)
        
        elif os.path.exists(dir + new_sub + '/' + filename + '/' + new_sub + '_0' + str(offframe) + '.jpg') ==True:
            return "0" + str(offframe)

        else:
            return -1

# Following functions in the below class are used to augment images using Numpy.
class Augmentation():

    # Translations are simple shifting of a picture in some direction. Below function accepts desired direction to move, amount of pixels for shift and behaviour of the patch that left empty when the image has been shifted. I prefer to roll patch that disappears on the edge to the other side of the image.
    @staticmethod
    def translate(img, shift=10, direction='right', roll=True):
        assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
        img1 = img.copy()

        if direction == 'right':
            right_slice = img1[:, -shift:].copy()
            img1[:, shift:] = img1[:, :-shift]
            if roll:
                img1[:,:shift] = np.fliplr(right_slice)

        if direction == 'left':
            left_slice = img1[:, :shift].copy()
            img1[:, :-shift] = img1[:, shift:]
            if roll:
                img1[:, -shift:] = left_slice

        if direction == 'down':
            down_slice = img1[-shift:, :].copy()
            img1[shift:, :] = img1[:-shift,:]
            if roll:
                img1[:shift, :] = down_slice

        if direction == 'up':
            upper_slice = img1[:shift, :].copy()
            img1[:-shift, :] = img1[shift:, :]
            if roll:
                img1[-shift:,:] = upper_slice
                
        return img1

    # The quite effective way to augment image is to rotate it a random degrees. The “empty” space in the corners has been filled with the mean of colours from the corner-patch.
    @staticmethod
    def rotate_img(img, angle, bg_patch=(15,15)):
        assert len(img.shape) <= 3, "Incorrect image shape"
        rgb = len(img.shape) == 3
        
        if rgb:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
        else:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])

        img = rotate(img, angle, reshape=False)
        mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
        img[mask] = bg_color

        return img

    # After applying translations and rotations it is helpful to add additional randomness in the augmented images by applying gaussian noise.
    @staticmethod
    def gaussian_noise(img, mean=5, sigma=1):
        noisy_img = img + np.random.normal(mean, sigma, img.shape)
        noisy_img_clipped = np.clip(noisy_img, 0, 255)  
        return noisy_img_clipped

    # Another interesting way to change original sample is to somehow distort it. As a simple example we can apply continuous shift of the rows or columns of our image guided by trigonometric functions (cosine or sinus). The resulting image would be “wavy” in horizontal or vertical directions. By tuning function parameters we can achieve required distortion power that produce different image with the same content. 

    # However, since facial microexpressions are very subtle, we won't be distorting the image during augmentation...it was found to obscure certain features that model learns from, after some experimentations.
    @staticmethod
    def distort(img, orientation='horizontal', func=np.sin, x_scale=0.02, y_scale=3):
        
        assert orientation[:3] in ['hor', 'ver'], "dist_orient should be 'horizontal'|'vertical'"
        assert func in [np.sin, np.cos], "supported functions are np.sin and np.cos"
        assert 0.00 <= x_scale <= 0.1, "x_scale should be in [0.0, 0.1]"
        assert 0 <= y_scale <= min(img.shape[0], img.shape[1]), "y_scale should be less then image size"

        img_dist = img.copy()
        
        def shift(x):
            return int(y_scale * func(np.pi * x * x_scale))

        for i in range(img.shape[orientation.startswith('ver')]):
            if orientation.startswith('ver'):
                img_dist[:, i] = np.roll(img[:, i], shift(i))
            else:
                img_dist[i, :] = np.roll(img[i, :], shift(i))
                
        return img_dist


# Returns 3 equally spaced frames between end_path1 and end_path2
def get_next_three(end_path1, end_path2):             
    next_three = []

    for i in range(len(end_path1)):
        if end_path1[i]=='_':
            last_underscore_pos1 = i
        
        elif end_path1[i]=='.':
            dot_pos1 = i

    number1 = end_path1[last_underscore_pos1+1:dot_pos1]

    for i in range(len(end_path2)):
        if end_path2[i]=='_':
            last_underscore_pos2 = i
        
        elif end_path2[i]=='.':
            dot_pos2 = i

    number2 = end_path2[last_underscore_pos2+1:dot_pos2]

    # Finding 3 equidistant numbers between number 1 and number 2
    x = int(number1)
    y = int(number2)
    add = (y-x)//4
    
    if number1[0]=='0' and number1[1]=='0':
        p1 = '00'+str(x+add)
        if os.path.exists(dir+end_path1[:last_underscore_pos1+1]+p1+'.jpg')==False:
            p1 = '0'+str(x+add)

        p2 = '00'+str(x+add*2)
        if os.path.exists(dir+end_path1[:last_underscore_pos1+1]+p2+'.jpg')==False:
            p2 = '0'+str(x+add*2)

        p3 = '00'+str(x+add*3)
        if os.path.exists(dir+end_path1[:last_underscore_pos1+1]+p3+'.jpg')==False:
            p3 = '0'+str(x+add*3)

    elif number1[0]=='0':
        p1 = '0'+str(x+add)
        if os.path.exists(dir+end_path1[:last_underscore_pos1+1]+p1+'.jpg')==False:
            p1 = str(x+add)

        p2 = '0'+str(x+add*2)
        if os.path.exists(dir+end_path1[:last_underscore_pos1+1]+p2+'.jpg')==False:
            p2 = str(x+add*2)

        p3 = '0'+str(x+add*3)
        if os.path.exists(dir+end_path1[:last_underscore_pos1+1]+p3+'.jpg')==False:
            p3 = str(x+add*3)

    else:
        p1 = str(x+add)
        p2 = str(x+add*2)
        p3 = str(x+add*3)

    next_three.extend([end_path1[:last_underscore_pos1+1]+p1+'.jpg', end_path1[:last_underscore_pos1+1]+p2+'.jpg', end_path1[:last_underscore_pos1+1]+p3+'.jpg'])
    return next_three


# Returns class-wise list, where each entry contains the path of onset, apex and offset frames of one instance in the dataset
def get_classes_list(df, type):
    if type=="multiclass":
        happiness = []
        surprise = []
        anger = []
        fear = []
        disgust = []
        contempt = []
        sadness = []

        for i in range(len(df)) :
            new_sub = get_subject(df.loc[i, "Subject"])

            new_onframe = get_on_frame(df, new_sub, str(df.loc[i, "Filename"]), df.loc[i, "Onset Frame"])
            new_apexframe = get_apex_frame(df, new_sub, str(df.loc[i, "Filename"]), df.loc[i, "Apex Frame"])
            new_offframe = get_off_frame(df, new_sub, str(df.loc[i, "Filename"]), df.loc[i, "Offset Frame"])

            if new_onframe==-1 or new_apexframe==-1 or new_offframe==-1:
                continue

            if df.loc[i, "Estimated Emotion"]=="Happiness":
                path1 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_onframe + '.jpg'
                path2 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_apexframe + '.jpg'
                path3 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_offframe + '.jpg'
                happiness.append((path1, path2, path3))
            
            elif df.loc[i, "Estimated Emotion"]=="Surprise":
                path1 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_onframe + '.jpg'
                path2 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_apexframe + '.jpg'
                path3 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_offframe + '.jpg'
                surprise.append((path1, path2, path3))
            
            elif df.loc[i, "Estimated Emotion"]=="Anger":
                path1 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_onframe + '.jpg'
                path2 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_apexframe + '.jpg'
                path3 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_offframe + '.jpg'
                anger.append((path1, path2, path3))

            elif df.loc[i, "Estimated Emotion"]=="Fear":
                path1 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_onframe + '.jpg'
                path2 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_apexframe + '.jpg'
                path3 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_offframe + '.jpg'
                fear.append((path1, path2, path3))

            elif df.loc[i, "Estimated Emotion"]=="Disgust":
                path1 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_onframe + '.jpg'
                path2 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_apexframe + '.jpg'
                path3 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_offframe + '.jpg'
                disgust.append((path1, path2, path3))

            elif df.loc[i, "Estimated Emotion"]=="Contempt":
                path1 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_onframe + '.jpg'
                path2 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_apexframe + '.jpg'
                path3 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_offframe + '.jpg'
                contempt.append((path1, path2, path3))
            
            elif df.loc[i, "Estimated Emotion"]=="Sadness":
                path1 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_onframe + '.jpg'
                path2 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_apexframe + '.jpg'
                path3 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_offframe + '.jpg'
                sadness.append((path1, path2, path3))

        return [happiness,surprise,anger,fear,disgust,contempt,sadness]
    
    elif type=="binary":
        positive = []
        negative = []
        for i in range(len(df)) :
            new_sub = get_subject(df.loc[i, "Subject"])

            new_onframe = get_on_frame(df, new_sub, str(df.loc[i, "Filename"]), df.loc[i, "Onset Frame"])
            new_apexframe = get_apex_frame(df, new_sub, str(df.loc[i, "Filename"]), df.loc[i, "Apex Frame"])
            new_offframe = get_off_frame(df, new_sub, str(df.loc[i, "Filename"]), df.loc[i, "Offset Frame"])

            if new_onframe==-1 or new_apexframe==-1 or new_offframe==-1:
                continue

            if df.loc[i, "Estimated Emotion"]=="Happiness":
                path1 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_onframe + '.jpg'
                path2 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_apexframe + '.jpg'
                path3 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_offframe + '.jpg'
                positive.append((path1, path2, path3))
            
            elif df.loc[i, "Estimated Emotion"]!="Surprise" and df.loc[i, "Estimated Emotion"]!="Other":
                path1 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_onframe + '.jpg'
                path2 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_apexframe + '.jpg'
                path3 = new_sub + '/' + str(df.loc[i, "Filename"]) + '/' + new_sub + '_' + new_offframe + '.jpg'
                negative.append((path1, path2, path3))

        return [positive,negative]
    
    else:
        raise ValueError("Invalid argument")


# Plots 2 plots after training, for both training and validation process -  epoch vs accuracy and epoch vs loss
def plot_training_graphs(model_fit):
    plt.plot(model_fit.history['accuracy'])
    plt.plot(model_fit.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Cross-validation'], loc='upper left')
    plt.show()

    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Cross-validation'], loc='upper left')
    plt.show()
