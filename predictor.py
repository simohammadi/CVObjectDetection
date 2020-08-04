from patcher import patcher
from spacer_cropper import spacer_cropper

import os
from keras.models import load_model
import cv2 as cv
from keras import models
from keras import optimizers
import glob
import numpy as np


class predictor:

    model = None
    crop_img = None
    res = list()

    def __init__(self, model_path, img_path, img_dim, mode):
        
        self.model = load_model(model_path)
        
        self.model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-5),
            metrics=['accuracy'])
        if mode == "trans":
            [top_res, bot_res] = self.transition(img_path, img_dim)
            self.res.append(top_res)
            self.res.append(bot_res)
        else:
            comp_class = self.component(img_path, img_dim)
            self.res.append(comp_class)
    
    def component(self, img_path, img_dim):
        sc = spacer_cropper(img_path)
        img = sc.cropped_img
        img = cv.resize(img, (img_dim, img_dim)).astype("float32")
        img /= 255
        img_dim = [1, img_dim, img_dim, 3]
        img = np.reshape(img, img_dim)

        return self.model.predict_classes(img)
        
    def transition(self, img_path, patch_dim):
        sc = spacer_cropper(img_path)
        img = sc.cropped_img
        
        pth = patcher(img_path, patch_dim)
        [patches, names] = pth.get_border_patches("patcher")
        indices = self.get_indices(names)
        weights = self.weights(indices)

        top_img = list()
        bot_img = list()

        size = len(indices)
        for i in range(0, size):
            if(indices[i][0] == 0):
                temp = patches[i].astype("float32")
                top_img.append(temp/255)
            else:
                temp = patches[i].astype("float32")
                bot_img.append(temp/255)
            


        top_res = self.classify(patch_dim, top_img, weights)
        bot_res = self.classify(patch_dim, top_img, weights)
        
        return top_res, bot_res
        
    def classify(self, patch_dim, images, weights):
        patch_dim = [1,patch_dim, patch_dim, 3]
        classes = list()
        res = 0
        size = len(images)
        for img, i in zip(images, range(0,size)):
            img = np.reshape(img, patch_dim)
            classes.append(self.model.predict_classes(img)[0][0])
            res += weights[i]*classes[i]
        return res #fix so this is average

    def weights(self, indices):
        [middle, max_val] = self.get_midpoints(indices)
        weights = list()
        increment = 1/(max_val/2)
        for i in range(0,max_val+1):
            if (len(middle) == 1):
                if(i<(max_val+1)/2):
                    weights.append(increment*i)
                else:
                    weights.append(weights[max_val-i])
            else:
                if(i<(max_val-1)/2):
                    weights.append(increment*i)
                elif(i <= (max_val+1)/2):
                    weights.append(1)
                else:
                    weights.append(weights[max_val-i])

        return weights
            
    def get_midpoints(self, indices):
        max_val = max(l[1] for l in indices)
        remainder = max_val % 2
        midpoint = int(max_val/2)
        middle = list()
        if(remainder != 0):
            middle.append(midpoint)
            middle.append(midpoint+1)
        else:
            middle.append(midpoint)
        return middle, max_val

    #indices is in form [y,x]
    def get_indices(self, names):
        indices = list()

        for name in names:
            indices.append(self.get_index(name))
        
        return indices

    #gives index in [y,x]
    def get_index(self, name):
        name = os.path.splitext(name)
        name = name[0].split("_")

        index = list()

        for sub in name:
            if(sub[0] == "y"):
                index.append(int(sub[1]))
            if(sub[0] == "x"):
                index.append(int(sub[1]))
        
        return index
              

def main():
    p = predictor("finetuned.h5", "/home/sina/Documents/abb/transition_pictures/with_spacer/good_candidate/DSC_4819.jpg")

if __name__ == "__main__":
    main()