import cv2
import numpy as np
from numpy import*
import pandas as pd
import sys
from PIL import Image
import glob, os
import numpy
import matplotlib.pyplot as plt
from operator import itemgetter


# original png file for id scan
# ORIGINAL_PATH = "./data/image_for_crop/original/"
# PREDICT_PATH = "./result/"
# SAVE_PATH = "./data/image_for_crop/cropped/"

def cropped_image(ORIGINAL_PATH, PREDICT_PATH, SAVE_PATH):
    train_images = [os.path.basename(x).replace(".png","") for x in glob.glob(os.path.join(ORIGINAL_PATH,'*.png'))]
    for id in train_images: 
        print(id)
    #predicted UNET
        image_test = cv2.imread(PREDICT_PATH+id+"_predict.png")
        height_croped, width_croped, _ = image_test.shape
        image_test = cv2.resize(image_test,(256,256))
        image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    # 256x256 png file
        image_test_real = cv2.imread(ORIGINAL_PATH+id+".png")
        image_test_real = cv2.resize(image_test_real,(256,256))
        image_test_real = cv2.cvtColor(image_test_real, cv2.COLOR_BGR2GRAY)
        valuexx = []
        cordinate_backbone = []
        for i in range(len(image_test_real[:,])):
            x = image_test_real[:,i]
            valuexx.append(mean([x]))

        for i in range(1,257):
            cordinatebb = (i,valuexx[i-1])
            if  50 <= i <=200 :
                cordinate_backbone.append(cordinatebb)


        cordinate_back_bone1 = max(cordinate_backbone,key=lambda item:item[1])
        

        cordinate_back_bone = round(cordinate_back_bone1[0])
    
        left_chanel = []
        right_chanel = []
        for i in range(1,255):
            
            xleft_image = image_test[i,0:cordinate_back_bone]  
            max_value = numpy.where(xleft_image == 255)
            
    
            if any(max_value) == True:    

                max_value_left = numpy.amax(max_value)   
                left_chanel.append((i,max_value_left))
            

            xright_image = image_test[i,cordinate_back_bone:255] 
            min_value = numpy.where(xright_image == 255)
        

            if any(min_value) == True:
                
                min_value_right = numpy.amin(min_value)
                right_chanel.append((i,min_value_right))
            
        if any(left_chanel) == True:
            left_chanel_final = max(left_chanel,key=lambda item:item[1])
        else:
            left_chanel_final = (cordinate_back_bone,0)

        if any(right_chanel) == True:
            right_chanel_final = min(right_chanel,key=lambda item:item[1])
            right_chanel_final = (right_chanel_final[0],right_chanel_final[1]+cordinate_back_bone)
        else:
            right_chanel_final = (cordinate_back_bone,255)


        ret,thresh1 = cv2.threshold(image_test,250,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh1, 1, 2)

        #left demention for crop
        for i in range(1, len(contours)+1):    
            dist = cv2.pointPolygonTest(contours[i-1],(left_chanel_final[1],left_chanel_final[0]),True) 

            if dist >= 0 :
                contours_final_left = contours[i-1]

        # 3 cordinate
        xmin_left = np.amin(contours_final_left, axis=0) 
        xmin_left_final = xmin_left[0,0]
        #print(xmin_left_final)

        test = np.flip(contours_final_left, 1)
        ymin_left = np.amin(test, axis=0) 
        ymin_left_final = ymin_left[0,1]
        #print(ymin_left_final)

        ymax_left = np.max(test, axis=0) 
        ymax_left_final = ymax_left[0,1]
        #print(ymax_left_final)


        #right demention for crop
        for i in range(1, len(contours)+1):    
            dist = cv2.pointPolygonTest(contours[i-1],(right_chanel_final[1],right_chanel_final[0]),True) 

            if dist >= 0 :
                contours_final_right = contours[i-1]

        # 3 cordinate
        xmax_right = np.amax(contours_final_right, axis=0) 
        xmin_right_final = xmax_right[0,0]
        #print(xmin_right_final)

        test2 = np.flip(contours_final_right, 1)
        ymin_right = np.amin(test2, axis=0) 
        ymin_right_final = ymin_right[0,1]
        #print(ymin_right_final)

        ymax_right = np.max(test2, axis=0) 
        ymax_right_final = ymax_right[0,1]
        #print(ymax_right_final)

        #crop
        a = max(abs(xmin_left_final-cordinate_back_bone),abs(xmin_right_final-cordinate_back_bone))

        b = min(ymin_left_final,ymin_right_final)
        c = max(ymax_left_final,ymax_right_final)

        h1 = b-10
        if h1<0:
            h1 = 0
        h2 = c+10
        if h2>255:
            h2 = 255

        w1 = cordinate_back_bone-a-10
        if w1<0:
            w1 = 0
        w2 = cordinate_back_bone+a+10
        if w2>255:
            w2 = 255

        if h1<0:
            h1 = 0
        #for png
    # main data 
        img_orrgi_full = ORIGINAL_PATH +id +'.png'
        img_orgi = cv2.imread(img_orrgi_full)
        height_og, width_og, channels_og = img_orgi.shape


    
    # height_croped, width_croped, channels_croped = image_test.shape
        
        rate_width = width_og / width_croped
        rate_height = height_og/height_croped

        hf1 = int(h1*rate_height)
        hf2 = int(h2*rate_height)
        wf1 = int(w1*rate_width)
        wf2 = int(w2*rate_width)
        
        crop_img = img_orgi[ hf1:hf2,wf1:wf2]


 # save
        new_id = SAVE_PATH + id +"_cropped.png"
        cv2.imwrite(new_id, crop_img)