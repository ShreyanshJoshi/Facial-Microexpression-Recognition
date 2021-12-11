import matplotlib.pyplot as plt
from pandas.core import frame
import numpy as np
from numpy import newaxis
import random
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.common import get_next_three

dir = "C:/Users/Shreyansh/Desktop/Microexpression Detection/SAMM/"

'''Returns 9 equidistant frames (including onset, apex and offset frames) between onset frame to offset frame.'''
def get_frames_3dimages_helper(i):      
    frames = []
    img_arr1 = img_to_array(load_img(dir + i[0], color_mode="grayscale", target_size=(128,128)))    # Onset frame

    next_three = get_next_three(i[0], i[1])         # 3 equidistant frames between onset and apex (excluding both)
    img_arr1_1 = img_to_array(load_img(dir + next_three[0], color_mode="grayscale", target_size=(128,128)))
    img_arr1_2 = img_to_array(load_img(dir + next_three[1], color_mode="grayscale", target_size=(128,128)))
    img_arr1_3 = img_to_array(load_img(dir + next_three[2], color_mode="grayscale", target_size=(128,128)))

    img_arr2 = img_to_array(load_img(dir + i[1], color_mode="grayscale", target_size=(128,128)))    # Apex frame

    next_three = get_next_three(i[1], i[2])         # 3 equidistant frames between apex and offset (excluding both)
    img_arr2_1 = img_to_array(load_img(dir + next_three[0], color_mode="grayscale", target_size=(128,128)))
    img_arr2_2 = img_to_array(load_img(dir + next_three[1], color_mode="grayscale", target_size=(128,128)))
    img_arr2_3 = img_to_array(load_img(dir + next_three[2], color_mode="grayscale", target_size=(128,128)))

    img_arr3 = img_to_array(load_img(dir + i[2], color_mode="grayscale", target_size=(128,128)))    # Offset frame

    frames.extend([img_arr1,img_arr1_1,img_arr1_2,img_arr1_3,img_arr2,img_arr2_1,img_arr2_2,img_arr2_3,img_arr3])
    return frames


'''Returns a list of frames wherein each frame has a new added dimension. This is essential to get the frames in a particular shape after concatenating them
depthwise.'''
def get_frames_3dimages(frames, i):
    frames.clear()
    frames = get_frames_3dimages_helper(i)
    frames[0] = frames[0][newaxis,:,:,:]
    frames[1] = frames[1][newaxis,:,:,:]
    frames[2] = frames[2][newaxis,:,:,:]
    frames[3] = frames[3][newaxis,:,:,:]
    frames[4] = frames[4][newaxis,:,:,:]
    frames[5] = frames[5][newaxis,:,:,:]
    frames[6] = frames[6][newaxis,:,:,:]
    frames[7] = frames[7][newaxis,:,:,:]
    frames[8] = frames[8][newaxis,:,:,:]

    return frames


'''Returns list of classes, with labels and the paths of corresponding 9 frames.'''
def get_classes_list_with_9frames_and_labels(classes, type):
    if type=="multiclass":
        happiness = []
        surprise = []
        anger = []
        fear = []
        disgust = []
        contempt = []
        sadness = []

        for i in classes[0]:
            next_three = get_next_three(i[0], i[1])             
            next_three1 = get_next_three(i[1], i[2])            
            happiness.append((0,(i[0],next_three[0],next_three[1],next_three[2],i[1],next_three1[0],next_three1[1],next_three1[2],i[2])))

        for i in classes[1]:
            next_three = get_next_three(i[0], i[1])
            next_three1 = get_next_three(i[1], i[2])
            surprise.append((1,(i[0],next_three[0],next_three[1],next_three[2],i[1],next_three1[0],next_three1[1],next_three1[2],i[2])))

        for i in classes[2]:
            next_three = get_next_three(i[0], i[1])
            next_three1 = get_next_three(i[1], i[2])
            anger.append((2,(i[0],next_three[0],next_three[1],next_three[2],i[1],next_three1[0],next_three1[1],next_three1[2],i[2])))

        for i in classes[3]:
            next_three = get_next_three(i[0], i[1])
            next_three1 = get_next_three(i[1], i[2])
            fear.append((3,(i[0],next_three[0],next_three[1],next_three[2],i[1],next_three1[0],next_three1[1],next_three1[2],i[2])))

        for i in classes[4]:
            next_three = get_next_three(i[0], i[1])
            next_three1 = get_next_three(i[1], i[2])
            disgust.append((4,(i[0],next_three[0],next_three[1],next_three[2],i[1],next_three1[0],next_three1[1],next_three1[2],i[2])))

        for i in classes[5]:
            next_three = get_next_three(i[0], i[1])
            next_three1 = get_next_three(i[1], i[2])
            contempt.append((5,(i[0],next_three[0],next_three[1],next_three[2],i[1],next_three1[0],next_three1[1],next_three1[2],i[2])))

        for i in classes[6]:
            next_three = get_next_three(i[0], i[1])
            next_three1 = get_next_three(i[1], i[2])
            sadness.append((6,(i[0],next_three[0],next_three[1],next_three[2],i[1],next_three1[0],next_three1[1],next_three1[2],i[2])))

        return [happiness, surprise, anger, fear, disgust, contempt, sadness]

    elif type=="binary":
        positive = []
        negative = []

        for i in classes[0]:
            next_three = get_next_three(i[0], i[1])             # 3 equidistant frames between onset and apex (excluding both)         
            next_three1 = get_next_three(i[1], i[2])            # 3 equidistant frames between apex and offset (excluding both)
            positive.append((0,(i[0],next_three[0],next_three[1],next_three[2],i[1],next_three1[0],next_three1[1],next_three1[2],i[2])))

        for i in classes[1]:
            next_three = get_next_three(i[0], i[1])            
            next_three1 = get_next_three(i[1], i[2])            
            negative.append((1,(i[0],next_three[0],next_three[1],next_three[2],i[1],next_three1[0],next_three1[1],next_three1[2],i[2])))

        return [positive, negative]

    else:
        raise ValueError("Invalid argument")


'''Divides the shuffled dataset into train (which is later divided into train and val), and test sets in a particular ratio and returns the shuffled sets.'''
def get_train_test_list(classes, type):
    if type=="multiclass":
        random.shuffle(classes[0])
        random.shuffle(classes[1])
        random.shuffle(classes[2])
        random.shuffle(classes[3])
        random.shuffle(classes[4])
        random.shuffle(classes[5])
        random.shuffle(classes[6])

        train_list = classes[0][:19]
        test_list = classes[0][19:]

        train_list.extend(classes[1][:10])
        test_list.extend(classes[1][10:])

        train_list.extend(classes[2][:41])
        test_list.extend(classes[2][41:])

        train_list.extend(classes[3][:5])
        test_list.extend(classes[3][5:])

        train_list.extend(classes[4][:5])
        test_list.extend(classes[4][5:])

        train_list.extend(classes[5][:7])
        test_list.extend(classes[5][7:])

        train_list.extend(classes[6][:3])
        test_list.extend(classes[6][3:])

        random.shuffle(train_list)
        random.shuffle(test_list)
        return [train_list, test_list]
    
    elif type=="binary":
        random.shuffle(classes[0])
        random.shuffle(classes[1])

        train_list = classes[0][:19]
        test_list = classes[0][19:]
        train_list.extend(classes[1][:62])
        test_list.extend(classes[1][62:])

        random.shuffle(train_list)
        random.shuffle(test_list)
        return [train_list, test_list]

    else:
        raise ValueError("Invalid argument")
        

'''Below function makes predictions on the test set and prints the accuracy. Furthermore in case of multiclass classification, it also prints the confusion matrix,
helping analyze better the predictions model has made and how it can be improved. Since, this function is used in case of majority voting policy, predictions are 
made on each of the 9 frames of an entry in the test set (in order from onset to offset). A particular entry in the test set is said to be correctly predicted by 
our model only if it correctly predicts the class of majority, of the 9 frames (at least 5 correct) that the entry has.'''
def test_predictions_2dcnn(test_list, model, type="binary"):
    correct=0
    wrong=0
    
    if type=="multiclass":
        test_pred = []
        test_labels = []

        for i in test_list:  
            count = 0
            temp = []      
            for j in i[1]:
                ff = np.array(img_to_array(load_img(dir + j, color_mode="grayscale", target_size=(128,128))))
                ff = np.expand_dims(ff, axis=0)
                result = model.predict(x=ff)
                classes = np.argmax(result,axis=1)

                temp.append(classes[0])
                # print("Predicted class: ", classes[0], end=' ')
                # print(" and the correct class: ",i[0])

                if classes[0]==i[0]:
                    count+=1
            
            if count>4:
                correct+=1
            else:
                wrong+=1

            test_pred.append(max(set(temp), key = temp.count))
            test_labels.append(i[0])

    elif type=="binary":
        for i in test_list:  
            count = 0      
            for j in i[1]:
                ff = np.array(img_to_array(load_img(dir + j, color_mode="grayscale", target_size=(128,128))))
                ff = np.expand_dims(ff, axis=0)
                result = model.predict(x=ff)

                if result[0][0]>0.5:
                    ans = 0
                else:
                    ans = 1

                if ans==i[0]:
                    count+=1
            
            if count>4:
                correct+=1
            else:
                wrong+=1
    
    else:
        raise ValueError("Invalid argument")
    
    print((correct)/(correct+wrong))

    if type=="multiclass":  
        matrix = confusion_matrix(test_labels, test_pred, normalize='true')
        display_labels = ['Happiness','Surprise','Anger','Fear','Disgust','Comptempt','Sadness']
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)

        fig, ax = plt.subplots(figsize=(17, 6))
        disp = disp.plot(include_values=True, cmap=plt.cm.Blues, ax=ax, xticks_rotation='horizontal')
        plt.show()
