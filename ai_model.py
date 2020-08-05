#Modules

import os
#Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from model_trainer import model_trainer 
from preproc_cable import preproc_cable
from preproc_transition import preproc_transition
from patcher import patcher
from spacer_cropper import spacer_cropper
from tester import tester
from predictor import predictor

from keras.models import load_model
import cv2 as cv
import sys

class ai_model:

    name = str()

    def __init__(self,mode, path_to_model_trans, path_to_model_comp, path_to_img):
        self.classify(path_to_model_trans, path_to_img, 150, "trans")
        res = self.classify(path_to_model_comp, path_to_img, 460, None)
        if (mode == "comp"):
            print(self.decoder(res)[2])
        elif(mode == "trans"):
            print(self.decoder(res)[0], self.decoder(res)[1])

    def classify(self, model_path, img_path, img_dim, mode):
        p = predictor(model_path, img_path, img_dim, mode)
        return p.res

    def decoder(self, res):
        human_res = list()
        if(res[0] > 0):
            human_res.append("transition top")
        else:
            human_res.append("no transition top")
        if (res[1] > 0):
            human_res.append("transition bot")
        else:
            human_res.append("no transition bot")
        if(res[2] == 0):
            human_res.append("8 components")
        elif(res[2] == 1):
            human_res.append("9 components")
        elif(res[2] == 2):
            human_res.append("12 components")
        elif(res[2] == 3):
            human_res.append("22 components")
        return human_res
             
def main():
    if(len(sys.argv) == 5):
        path_to_model_trans = sys.argv[1]
        path_to_model_comp = sys.argv[2]
        path_to_img = sys.argv[3]
        mode = sys.argv[4]
        am = ai_model(path_to_model_trans, path_to_model_comp, path_to_img)
    elif(len(sys.argv) == 3):
        path_to_model_trans = "finetuned_trans.h5"
        path_to_model_comp = "finetuned_cablecable.h5"
        path_to_img = sys.argv[1]
        mode = sys.argv[2]
        am = ai_model(mode,path_to_model_trans, path_to_model_comp, path_to_img)
    else:
        print("Something went wrong")
    
    
        

if __name__ == "__main__":
    main()
